curl -X POST https://api.modelarts-maas.com/v2/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${HUAWEI_API_KEY}" \
  -d '{
    "model": "deepseek-v4.1-flash-1017",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello! Introduce yourself in one sentence."
      }
    ]
  }'
