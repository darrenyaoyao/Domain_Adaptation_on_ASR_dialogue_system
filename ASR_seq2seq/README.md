Run Code:
  python chatbot.py --pretrain (pretrain original seq2seq, --schedule_sampling for scheduled sampling learning)
  python chatbot.py --train_encoder (train MSE loss between original and ASR encoder)
  python chatbot.py --fine_tune (fine-tune ASR seq2seq, --schedule_sampling for scheduled sampling learning)
  python chatbot.py --test (evaluate the ppx of original testing data, --use_asr for ASR testing data)
  python chatbot.py --decode (interactive decoding mode, --use_asr for ASR input)
