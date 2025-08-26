pip install -r requirements.txt

```
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5
```

ImportError: libcudart.so.12: cannot open shared object file: No such file or directory 의 에러가 났었는데 torch의 버전을 upgrade하고 다시 pip -r requirements.txt를 깔았더니 다시 작동 됨. 아마도 추측되는 상황은 기존의 docker의 세팅에서 compile된 torch를 부르지 못하는데 upgrade를 하면 graphic card와 version sync가 깨지게 되고 다시 pip -r requiremnets.txt를 깔게 되면 정장작동하는 것으로 보임

```
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
}'
```

INFO 08-25 05:45:10 [gpu_model_runner.py:1186] Model loading took 15.6271 GB and 4.171168 seconds
|    3   N/A  N/A   1452779      C   /opt/conda/bin/python3                    42886MiB |
 16%|██████                                | 608/3787 [36:44<2:03:24,  2.33s/it]