import http.client
import json


def sendWarn(task_id, category):
    try:
        # 登录请求
        conn = http.client.HTTPConnection("106.12.166.86", 8181)

        payload = json.dumps({
            "accessKeyId": "1275A3829C51A903",
            "accessKeySecret": "48B3B3B91275A3829C51A903EE2D30EE"
        })
        headers = {
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/system/auth/access/login", payload, headers)

        res = conn.getresponse()
        login_data = res.read().decode("utf-8")
        print("登录响应:", login_data)

        # 解析登录响应，获取 access_token
        login_json = json.loads(login_data)
        if login_json.get("code") == 200:
            access_token = login_json["data"]["access_token"]
        else:
            print("登录失败，无法获取 access_token")
            return None
    except http.client.HTTPException as http_err:
        print(f"登录请求时发生HTTP错误: {http_err}")
        return None
    except UnicodeDecodeError as decode_err:
        print(f"登录响应解码时发生错误: {decode_err}")
        return None
    except Exception as err:
        print(f"登录请求时发生未知错误: {err}")
        return None

    try:
        # 获取任务信息请求
        conn = http.client.HTTPConnection("106.12.166.86", 8181)

        payload = ''
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        url = f"/ai/AIModel/discernShout/{category}/{task_id}"
        conn.request("GET", url, payload, headers)
        res = conn.getresponse()
        task_data = res.read().decode("utf-8")
        print("任务信息响应:", task_data)
        return task_data
    except http.client.HTTPException as http_err:
        print(f"获取任务信息请求时发生HTTP错误: {http_err}")
        return None
    except UnicodeDecodeError as decode_err:
        print(f"任务信息响应解码时发生错误: {decode_err}")
        return None
    except Exception as err:
        print(f"获取任务信息请求时发生未知错误: {err}")
        return None


