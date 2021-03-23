import SessionState
import streamlit as st
import pandas as pd
from fitbert import FitBert
from transformers import (
    BlenderbotSmallTokenizer, 
    BlenderbotSmallForConditionalGeneration,
    MobileBertForMaskedLM,
    MobileBertTokenizer,
    pipeline
)

@st.cache(allow_output_mutation=True, max_entries=1)
def load_Blenderbot(Bbot_PATH):
    BbotModel = BlenderbotForConditionalGeneration.from_pretrained(Bbot_PATH)
    BbotTokenizer = BlenderbotTokenizer.from_pretrained(Bbot_PATH)
    return BbotModel, BbotTokenizer

@st.cache(allow_output_mutation=True, max_entries=1)
def load_BlenderbotSmall(Bbot_PATH):
    BbotModel = BlenderbotSmallForConditionalGeneration.from_pretrained(Bbot_PATH)
    BbotTokenizer = BlenderbotSmallTokenizer.from_pretrained(Bbot_PATH)
    return BbotModel, BbotTokenizer

@st.cache(allow_output_mutation=True, max_entries=1)
def load_MobileBERT(MBERT_PATH):
    MBERTmodel = MobileBertForMaskedLM.from_pretrained(MBERT_PATH)
    MBERTtokenizer = MobileBertTokenizer.from_pretrained(MBERT_PATH)
    return MBERTmodel, MBERTtokenizer

def main():
    session_state = SessionState.get(user_chat_log=[], bot_reply_log=[])

    mode = st.sidebar.radio("Mode: ", options=['Chat', 'English_solver'])

    # mode = st.radio("Mode: ", options=['Chat', 'TOEIC_part5'])

    # Reserve space for chatlog
    chatlog_holder = st.empty()
    
    if mode == 'Chat':
        # Load BlenderbotSmall chạy local
        # Tải thêm file từ
        # https://huggingface.co/facebook/blenderbot_small-90M/tree/main
        # vào folder BlenderbotSmall

        # Bbot_PATH = 'facebook/blenderbot-400M-distill'
        # Bbot_PATH = 'facebook/blenderbot_small-90M'
        # Bbot_PATH = './Blenderbot'
        Bbot_PATH = './BlenderbotSmall'

        BbotModel, BbotTokenizer = load_BlenderbotSmall(Bbot_PATH)

        text = st.text_input("You:")
        if len(text) != 0:
            inputs = BbotTokenizer([text], return_tensors='pt')
            reply_ids = BbotModel.generate(**inputs)
            bot_reply = BbotTokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            session_state.user_chat_log.append(text)
            session_state.bot_reply_log.append(bot_reply)
            df = pd.DataFrame({'You': session_state.user_chat_log, 'Bot': session_state.bot_reply_log})
            with chatlog_holder:
                st.table(df.tail(5))
            with st.beta_expander("Full chat history"):
                st.table(df)


    elif mode == 'English_solver':
        # Load MobileBERT chạy local
        # Tải thêm file từ
        # https://huggingface.co/google/mobilebert-uncased/tree/main
        # vào folder MobileBERT
        MBERT_PATH = './MobileBERT'
        MBERTmodel, MBERTtokenizer = load_MobileBERT(MBERT_PATH)
        fb = FitBert(model=MBERTmodel, tokenizer=MBERTtokenizer)

        num_choices = st.sidebar.slider(label="Number of choices", min_value=0, max_value=4)

        st.sidebar.markdown("""
        0: Fill in the blank

        1: Grammatical conjugation

        2-4: Sentence completion""")

        if num_choices == 0:
            # Fill in the blank
            question = st.text_input(label='Sentence:')
            question = question.replace('_', '[MASK]')

            mlm = pipeline('fill-mask', model=MBERTmodel, tokenizer=MBERTtokenizer)
            result = mlm(question)[0]['token_str'].replace(' ','')

            session_state.user_chat_log.append(question.replace('[MASK]', '_'))
            session_state.bot_reply_log.append(result)
            df = pd.DataFrame({'You': session_state.user_chat_log, 'Bot': session_state.bot_reply_log})
            with chatlog_holder:
                st.table(df.tail(5))
            with st.beta_expander("Full chat history"):
                st.table(df)

        elif num_choices == 1:
            # Grammatical conjugation
            question = st.text_input(label='Sentence:')
            choices=[st.text_input(label='Word needs to conjugate')]
            question = question.replace('_', '***mask***')
            bot_choice = fb.fitb(question, options=choices)

            session_state.user_chat_log.append(question.replace('***mask***', '_'))
            session_state.bot_reply_log.append(bot_choice)
            df = pd.DataFrame({'You': session_state.user_chat_log, 'Bot': session_state.bot_reply_log})
            with chatlog_holder:
                st.table(df.tail(5))
            with st.beta_expander("Full chat history"):
                st.table(df)
            
        else:
            # Sentence completion
            labels=['A.', 'B.', 'C.', 'D.']
            choices=[]
            question = st.text_input(label='Question:')
            for i in range(num_choices):
                choices.append(st.text_input(label=labels[i]))
            
            if st.button("Solve") and len(question) != 0 and len(choices[0]) != 0 and len(choices[1]) != 0:
                question = question.replace('_', '***mask***')
                bot_choice = fb.rank(question, options=choices)[0]

                session_state.user_chat_log.append(question.replace('***mask***', '_'))
                session_state.bot_reply_log.append(bot_choice)
                df = pd.DataFrame({'You': session_state.user_chat_log, 'Bot': session_state.bot_reply_log})
                with chatlog_holder:
                    st.table(df.tail(5))
                with st.beta_expander("Full chat history"):
                    st.table(df)
                

if __name__ == "__main__":
    main()