1) ```curl -L -o data/vocab.json  https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json```
2) ```curl -L -o data/merges.txt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt```
3) ```curl -L -C -   ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"}   -o data/v1-5-pruned-emaonly.ckpt   https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt```
