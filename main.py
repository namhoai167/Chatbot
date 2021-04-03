import SessionState
import streamlit as st
import pandas as pd
from fitbert import FitBert
from transformers import (
    BlenderbotSmallTokenizer, 
    BlenderbotSmallForConditionalGeneration,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    MobileBertForMaskedLM,
    MobileBertTokenizer,
    pipeline
)

@st.cache(allow_output_mutation=True, max_entries=1)
def load_Blenderbot(Bbot_PATH):
    BbotModel = BlenderbotForConditionalGeneration.from_pretrained(Bbot_PATH)
    BbotTokenizer = BlenderbotTokenizer.from_pretrained(Bbot_PATH)
    return BbotModel, BbotTokenizer

# @st.cache(allow_output_mutation=True, max_entries=1)
# def load_BlenderbotSmall(Bbot_PATH):
#     BbotModel = BlenderbotSmallForConditionalGeneration.from_pretrained(Bbot_PATH)
#     BbotTokenizer = BlenderbotSmallTokenizer.from_pretrained(Bbot_PATH)
#     return BbotModel, BbotTokenizer

# @st.cache(allow_output_mutation=True, max_entries=1)
# def load_MobileBERT(MBERT_PATH):
#     MBERTmodel = MobileBertForMaskedLM.from_pretrained(MBERT_PATH)
#     MBERTtokenizer = MobileBertTokenizer.from_pretrained(MBERT_PATH)
#     return MBERTmodel, MBERTtokenizer

@st.cache(allow_output_mutation=True, max_entries=1)
def load_ELECTRAsmall(ELETRA_PATH):
    ELECTRAmodel = ElectraForMaskedLM.from_pretrained(ELECTRA_PATH)
    ELECTRAtokenizer = ElectraTokenizer.from_pretrained(ELECTRA_PATH)
    return ELECTRAmodel, ELECTRAtokenizer

def chat_log(user_log, bot_log, chatlog_holder):
    df = pd.DataFrame({'You': user_log, 'Bot': bot_log})
    with chatlog_holder:
        st.table(df.tail(5))
    with st.beta_expander("Full chat history"):
        st.table(df)

def main():
    session_state = SessionState.get(user_chat_log=[], bot_reply_log=[])

    mode = st.sidebar.radio("Mode: ", options=['Chat', 'TOEIC_part5', 'TOEIC_part6'])

    # mode = st.radio("Mode: ", options=['Chat', 'TOEIC_part5'])

    # Reserve space for chatlog
    chatlog_holder = st.empty()
    
    if mode == 'Chat':
        # Load BlenderbotSmall chạy local
        # Tải thêm file từ
        # https://huggingface.co/facebook/blenderbot_small-90M/tree/main
        # vào folder BlenderbotSmall
        # Bbot_PATH = './blenderbot-400M-distill'
        # Bbot_PATH = './blenderbot_small-90M'

        # Chạy trên server streamlit thì thay path
        Bbot_PATH = 'facebook/blenderbot-400M-distill'
        # Bbot_PATH = 'facebook/blenderbot_small-90M'
        
        # BbotModel, BbotTokenizer = load_BlenderbotSmall(Bbot_PATH)
        BbotModel, BbotTokenizer = load_Blenderbot(Bbot_PATH)

        text = st.text_input("You:")
        if len(text) != 0:
            inputs = BbotTokenizer([text], return_tensors='pt')
            reply_ids = BbotModel.generate(**inputs)
            bot_reply = BbotTokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            session_state.user_chat_log.append(text)
            session_state.bot_reply_log.append(bot_reply)
            chat_log(session_state.user_chat_log, session_state.bot_reply_log, chatlog_holder)


    elif mode == 'TOEIC_part5':
        # Load ELECTRA small chạy local
        # Tải thêm file từ
        # https://huggingface.co/google/electra-small-generator/tree/main
        # vào folder electra-small-generator
        # ELECTRA_PATH = './electra-small-generator'

        # Chạy trên server streamlit thì thay path
        ELECTRA_PATH = 'google/electra-small-generator'

        ELECTRAmodel, ELECTRAtokenizer = load_ELECTRAsmall(ELECTRA_PATH)
        fb = FitBert(model=ELECTRAmodel, tokenizer=ELECTRAtokenizer)

        num_choices = st.sidebar.slider(label="Number of choices", min_value=0, max_value=4)

        st.sidebar.markdown("""
        0: Fill in the blank \\
        1: Grammatical conjugation \\
        2-4: Sentence completion""")

        if num_choices == 0:
            # Fill in the blank
            question = st.text_input(label='Sentence:')
            if len(question) != 0:
                session_state.user_chat_log.append(question)

                question = question.replace('_', '[MASK]')
    
                mlm = pipeline('fill-mask', model=ELECTRAmodel, tokenizer=ELECTRAtokenizer)
                result = mlm(question)[0]['token_str'].replace(' ','')
    
                session_state.bot_reply_log.append(result)
                chat_log(session_state.user_chat_log, session_state.bot_reply_log, chatlog_holder)

        elif num_choices == 1:
            # Grammatical conjugation
            question = st.text_input(label='Sentence:')
            choices=[st.text_input(label='Word needs to conjugate')]
            if st.button("Conjugate") and len(question) != 0 and choices:
                session_state.user_chat_log.append(question)

                question = question.replace('_', '***mask***')
                bot_choice = fb.fitb(question, options=choices)
    
                session_state.bot_reply_log.append(bot_choice)
                chat_log(session_state.user_chat_log, session_state.bot_reply_log, chatlog_holder)
            
            
        else:
            # Sentence completion
            labels=['A.', 'B.', 'C.', 'D.']
            choices=[]
            question = st.text_input(label='Question:')
            for i in range(num_choices):
                choices.append(st.text_input(label=labels[i]))
            
            if st.button("Solve") and len(question) != 0 and len(choices[0]) != 0 and len(choices[1]) != 0:
                session_state.user_chat_log.append(question)

                question = question.replace('_', '***mask***')
                bot_choice = fb.rank(question, options=choices)[0]

                session_state.bot_reply_log.append(bot_choice)
                chat_log(session_state.user_chat_log, session_state.bot_reply_log, chatlog_holder)
    
    elif mode == 'TOEIC_part6':
        # Load ELECTRA small chạy local
        # ELECTRA_PATH = './electra-small-generator'
        
        # Chạy trên server streamlit thì thay path
        ELECTRA_PATH = 'google/electra-small-generator'

        ELECTRAmodel, ELECTRAtokenizer = load_ELECTRAsmall(ELECTRA_PATH)
        fb = FitBert(model=ELECTRAmodel, tokenizer=ELECTRAtokenizer)
        question = []
        labels=['Question 1.', 'Question 2.', 'Question 3.', 'Question 4.']
        paragraph = st.text_input(label='Paragraph:')
        for i in range(4):
            question.append(st.text_input(label=labels[i]))
            question[i] = question[i].split('/')

        st.write(question)

        bot_choices = []
        
        if st.button("Solve") and len(paragraph) != 0:
            session_state.user_chat_log.append(paragraph)
            paragraph = paragraph.replace('_', '***mask***')
            splited_paragraph = paragraph.split()
            mask_indices = [i for i, item in enumerate(splited_paragraph) if '***mask***' in item]
            

            if len(paragraph) < len(mask_indices):
                length = len(paragraph)
            else:
                length = len(mask_indices)

            for i in range(length):
                mask_idx = mask_indices.pop(0)
                one_masked_paragraph = ' '.join([word for idx, word in enumerate(splited_paragraph) if idx not in mask_indices])
                st.write(one_masked_paragraph)
                bot_choices.append(fb.rank(sent=one_masked_paragraph, options=question[i])[0])
                splited_paragraph[mask_idx] = bot_choices[i]

            session_state.bot_reply_log.append(bot_choices)
            chat_log(session_state.user_chat_log, session_state.bot_reply_log, chatlog_holder)

                

if __name__ == "__main__":
    main()