import http.client
import json

conn = http.client.HTTPSConnection("106.12.166.86", 8181)
payload = json.dumps({
   "accessKeyId": "3D811D34CC7C958D",
   "accessKeySecret": "CAB7763C3D811D34CC7C958DA2E94674"
})
headers = {
   'Content-Type': 'application/json'
}
conn.request("POST", "/system/auth/access/login", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))