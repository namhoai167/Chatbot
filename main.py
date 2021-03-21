import streamlit as st
import pandas as pd
# from fitbert import FitBert
from transformers import (
    BlenderbotSmallTokenizer, 
    BlenderbotSmallForConditionalGeneration,
    MobileBertForMaskedLM,
    MobileBertTokenizer
)

def main():
    user_chat_log = []
    bot_reply_log = []
    
    mode = st.radio("Mode: ", options=['Chat', 'TOEIC_part5'])
    
    while mode == 'Chat':
        # Load Blenderbot chạy local
        # Tải file weight pytorch_model.bin từ 
        # https://huggingface.co/facebook/blenderbot_small-90M/tree/main
        # vào folder có tên Bbot_PATH
        # Tránh nhầm với file weight tf_model.h5 của tensorflow
        Bbot_PATH = './BlenderbotSmall'
        BbotModel = BlenderbotSmallForConditionalGeneration.from_pretrained(Bbot_PATH, local_files_only=True)
        BbotTokenizer = BlenderbotSmallTokenizer.from_pretrained(Bbot_PATH, local_files_only=True)

        # Reserve space for chatlog
        chatlog_holder = st.empty()

        text = st.text_input("You:")
        try:
            if len(text) > 1:
                inputs = BbotTokenizer([text], return_tensors='pt')
                reply_ids = BbotModel.generate(**inputs)
                bot_reply = BbotTokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

                user_chat_log.append(text)
                bot_reply_log.append(bot_reply)
                df = pd.DataFrame({'You': user_chat_log, 'Bot': bot_reply_log})

                with chatlog_holder:
                    st.table(df.tail(5))

                with st.beta_expander("Full chat history"):
                    st.table(df)
                    
        except:
            raise

    else:
        # Load MobileBERT
        # MBERT_PATH = './Chatbot/MobileBERT'
        # MBERTmodel = MobileBertForMaskedLM.from_pretrained(MBERT_PATH)
        # MBERTtokenizer = MobileBertTokenizer.from_pretrained(MBERT_PATH)
        # fb = FitBert(model=MBERTmodel, tokenizer=MBERTtokenizer)
        st.write('ok')
    


if __name__ == "__main__":
    main()