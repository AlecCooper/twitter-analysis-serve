rm $(pwd)/model-store/twitter-sentiment.mar
torch-model-archiver --model-name twitter-sentiment --version 1.0 --handler $(pwd)/model_handler.py --serialized-file $(pwd)/model/pytorch_model.bin --extra-files "$(pwd)/model/config.json,$(pwd)/model/vocab.txt"
mv $(pwd)/twitter-sentiment.mar ./model-store/twitter-sentiment.mar