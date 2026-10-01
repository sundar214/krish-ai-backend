import http.client
import json

conn = http.client.HTTPSConnection("api.kie.ai")
payload = json.dumps({
   "model": "gpt-5-6-sol",
   "input": [
      {
         "role": "user",
         "content": [
            {
               "type": "input_text",
               "text": "What is in this image?"
            },
            {
               "type": "input_image",
               "image_url": "https://file.aiquickdraw.com/custom-page/akr/section-images/1759055072437dqlsclj2.png"
            }
         ]
      }
   ],
   "tools": [
      {
         "type": "web_search"
      }
   ],
   "reasoning": {
      "effort": "high"
   }
})
headers = {
   'Authorization': 'Bearer <token>',
   'Content-Type': 'application/json'
}
conn.request("POST", "/codex/v1/responses", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))