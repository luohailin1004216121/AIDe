
import enum
from yaml.tokens import AnchorToken
import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):
	# return positive, negative label smoothing BCE targets
	return 1.0 - 0.5 * eps, 0.5* eps

class BCEBlurWithLogitsLoss(nn.Module):
	# BCEwithLogitLoss() with reduced missing label effects.
	def __init__(self, alpha=0.05):
		super(BCEBlurWithLogitsLoss, self).__init__()
		self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
		self.alpha = alpha

	def forward(self, pred, true):
		loss = self.loss_fcn(pred, true)
		pred = torch.sigmoid(pred)
		dx = pred - true
		alpha_factor = 1 - torch.exp((dx-1) / (self.alpha + 1e-4))
		loss *= alpha_factor
		return loss.mean()

class FocalLoss(nn.Module):
	# Wraps focal loss around existinf loss_fcn() i.e i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
	def __init__(self, loss_fcn, gamma=1.5, alpha = 0.25):
		super(FocalLoss, self).__init__()
		self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = loss_fcn.reduction
		self.loss_fcn.reduction = 'none' # rewuired to apply FL to each element

	def forward(self, pred, true):
		loss = self.loss_fcn(pred, true)

		pred_prob = torch.sigmoid(pred)
		p_t = true * pred_prob + (1- true) *(1- pred_prob)
		alpha_factor = true * self.alpha + (1- true) *(1-self.alpha)
		modulating_factor = (1.0-p_t) ** self.gamma
		loss *= alpha_factor * modulating_factor

		if self.reduction =='mean':
			return loss.mean()

		elif self.reduction =='sum':
			return loss.sum()
		
		else:
			return loss


class QFocalLoss(nn.Module):
	# Wraps Quality focal loss around existinf loss_fcn() i.e i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
	def __init__(self, loss_fcn, gamma=1.5, alpha = 0.25):
		super(FocalLoss, self).__init__()
		self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = loss_fcn.reduction
		self.loss_fcn.reduction = 'none' # rewuired to apply FL to each element

	def forward(self, pred, true):
		loss = self.loss_fcn(pred, true)

		pred_prob = torch.sigmoid(pred)
		alpha_factor = true * self.alpha + (1- true) *(1-self.alpha)
		modulating_factor = torch.abs(true- pred_prob) ** self.gamma
		loss *= alpha_factor * modulating_factor

		if self.reduction =='mean':
			return loss.mean()
		elif self.reduction =='sum':
			return loss.sum()		
		else:
			return loss


class ComputeLoss:
	# Compute losses
	def __init__(self, model, autobalance=False):
		self.sort_obj_iou = False
		device = next(model.parameters()).device   # to get the device
		h = model.hyp  # hyperparameters

		# Define criteria
		BCEcls = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([h['cls_pw']],device=device))
		BCEobj = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([h['obj_pw']],device=device))

		# class label smoothening
		self.cp, self.cn = smooth_BCE(eps = h.get('label_smoothing',0.0))
		# positive, negative BCE targets

		# Focal loss
		g = h['fl_gamma']  # focal loss gamma
		if g > 0:
			BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

		det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
		self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
		self.ssi = list(det.stride).index(16) if autobalance else 0 # stride 16 index
		self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
		for k in 'na', 'nc', 'nl', 'anchors':
			setattr(self, k, getattr(det, k))

	def __call__(self, p, targets):  # predictions, targets
		device = targets.device
		lcls, lbox, lobj = torch.zeros(1,device=device), torch.zeros(1,device=device), torch.zeros(1,device=device)
		tcls, tbox, indices, anchors = self.build_ragets(p,targets)

		# Losses
		for i, pi in enumerate(p):  # leyer index, layer predictions
			b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
			tobj = torch.zeros_like(pi[..., 0], device = device) # target obj

			n = b.shape[0] # number of targets
			if n:
				ps = pi[b, a, gj, gi]  # prediction subset correcposing to targets

				# Regression
				pxy = ps[:, :2].sigmoid() * 2. - 0.5
				pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
				pbox = torch.cat((pxy, pwh), 1) # predicted box
				iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
				lbox += (1.0 - iou).mean()

				# Objectness
				score_iou = iou.detach().clamp(0).type(tobj.dtype)
				if self.sort_obj_iou:
					sort_id = torch.argsort(score_iou)
					b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
				tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou

				# Classification

				if self.nc > 1:
					t = torch.full_like(ps[:, 5:], self.cn, device= device)
					t[range(n), tcls[i]] = self.cp
					tcls += self.BCEcls(ps[:, 5:], t)  # BCE

			obji = self.BCEobj(pi[..., 4], tobj)
			lobj += obji * self.balance[i]  # obj loss

			if self.autobalance:
				self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

		if self.autobalance:
			self.balance = [x / self.balance[self.ssi]] for x in self.balance]
		lbox *= self.hyp['box']
		lobj *= self.hyp['obj']
		lcls *= self.hyp['cls']

		bs = tobj.shape[0]  # batch size

		return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()	





	def build_targets(self, p, targets):
		# Build targets for computer_loss(), input targets(image, class,x,y,w,h)
		na, nt = self.na, targets.shape[0]  # number of anchors, targets
		tcls, tbox, indices, anch = [], [], [], []
		gain = torch.ones(7, device = targets.device)  # normalized to gridspace gain
		ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
		targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

		g = 0.5 #bias
		off = torch.tensor([[0, 0],
							[1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
							# [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
							], device=targets.device).float() * g  # offsets

		for i in range(self.nl):
			anchors = self.anchors[i]
			gain[2:6] = torch.tensor(p[i].shape)[[3,2,3,2]] #xyxy gain

			# Match targets to anchors
			t = targets * gain
			if nt:
				# Matches
				r = t[:, :, 4:6] / anchors[:, None] # wh ratio
				j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
				# j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
				t = t[j]  # filter

				# Offsets
				gxy = t[:, 2:4]  # grid xy
				gxi = gain[[2, 3]] - gxy  # inverse
				j, k = ((gxy % 1. < g) & (gxy > 1.)).T
				l, m = ((gxi % 1. < g) & (gxi > 1.)).T
				j = torch.stack((torch.ones_like(j), j, k, l, m))
				t = t.repeat((5, 1, 1))[j]
				offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
			else:
				t = targets[0]
				offsets = 0

			# Define
			b, c = t[:, :2].long().T # image, class
			gxy = t[:, 2:4]  # grid xy
			gwh = t[:, 4:6]  # grid wh
			gij = (gxy- offsets).long()
			gi, gj = gij.T

			# Append
			a = t[:, 6].long()  # anchor indices
			indices.append((b, a, gj.clamp_(0, gain[3] -1), gi.clamp_(0, gain[2] -1)))  # image, anchor, grid indices
			tbox.append(torch.cat((gxy - gij, gwh), 1)) # box
			anch.append(anchors[a])
			tcls.append(c)

		return tcls, tbox, indices, anch





		

