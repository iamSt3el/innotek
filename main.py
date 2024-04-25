import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from haystack.components.websearch import SerperDevWebSearch
from haystack.utils import Secret
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from urllib.parse import urlparse

#from langchain_anthropic import ChatAnthropic

#from transformers import VitsModel, AutoTokenizer,set_seed
#import torch
#import scipy
#import base64
#import os


# Call the function to clear the terminal
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyAlloUClKdRegH-pERfnmdrotLDL2HXIDQ",
                                 safety_settings= {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                   HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                   HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,})
    



#def get_voice(text):
#    model1 = VitsModel.from_pretrained("facebook/mms-tts-eng")
#    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
#    os.system("clear")
#    set_seed(555)
#    inputs = tokenizer(text, return_tensors="pt")
#    print("Generating....")
#    with torch.no_grad():
#        output = model1(**inputs).waveform
#    print("Done.")

    #saving the wav file
#    waveform_numpy = output.detach().numpy()[0]

#    print("Playing....")
#    scipy.io.wavfile.write("techno.wav", rate=model1.config.sampling_rate, data=waveform_numpy)



def is_pdf(url):
    parsed_url = urlparse(url)
    file_extension = parsed_url.path.split('.')[-1].lower()
    return file_extension == 'pdf'

def llm(prompt):
    response = model.invoke(prompt)
    return response.content

    #llm = ChatAnthropic(anthropic_api_key="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",
    #                   model="claude-2.1")
    
    #response = llm.invoke(prompt)
    #return response.content



def action_agent(user_question, hist):
    prompt_template = """
    You are an AI legal conversational assistant JURIS to provide information related to law, legal matters, and criminal issues related to india. 

For each user query, you have three options:

1. If you have sufficient knowledge to directly answer the legal question, provide a clear and detailed response, citing relevant laws, cases, or legal principles when applicable. Maintain objectivity and avoid giving advice that favors any particular side.

2. If you lack the necessary information to comprehensively address the query, output just the phrase "0" (zero). This will indicate that you need to access internet resources and external legal databases to research the topic more thoroughly before providing a complete answer.

3. You are also a conversational chatbot respond to users questions in a humane way, also understand the question based on history of conversation.

Always be professional, ethical, and empathetic in your language.
current date: {date}

You have access to the conversation history to maintain context:
{history}


When ready, analyze the user's new question:
{question}

Then either provide a direct legal explanation drawing from your knowledge base, responds users question in a humane way or just output "0" when you need help from external data.
    """

    prompt = prompt_template.format(question = user_question, history = hist, date = "24-April-2024")

    response = llm(prompt)

    return response


def google_search(user_question, hist):
    prompt_template = """Analyze the given user question and understand the intent behind it. 

                        Convert this intent into an effective Google search query that could potentially provide relevant information to address the user's question.

                        Your goal is to output a suitable Google search query, not to actually perform the search or provide an answer.
                        
                        current date: {date}
                        You have access to the conversation history for context:
                        {history}
                        
                        The user's current question is:
                        {question}

                        Based on the question, provide only the corresponding Google search query you would use to find relevant information to address the user's intent. Do not output any other text.
    """

    prompt = prompt_template.format(question = user_question, history = hist, date = "24-April-2024")

    response = llm(prompt)
    print(response)
    return response


def user_input(user_question, hist):
    prompt_template_1 = """You are a legal conversational AI assistant named JURIS trained to provide general legal information to users related to india,  within the scope of your knowledge base in a conversational manner. Your goals are:

                            1. Thoroughly understand each user's legal query by asking clarifying follow-up questions as needed.

                            2. Provide relevant legal information from authoritative sources like statutes, cases, and publications. Cite sources where applicable.

                            3. Explain legal concepts plainly while maintaining formal, professional language. 

                            4. Remind users your responses do not constitute formal legal advice. Encourage consulting lawyers for complex matters.

                            5. Maintain objectivity. Present different perspectives neutrally without favoring any side.

                            6. Uphold ethics like client confidentiality, avoiding conflicts of interest, and acting competently within abilities.

                            7. If outside expertise, politely suggest the user consult a qualified legal professional.

                            8. Access external knowledge from the internet to supplement your knowledge base when relevant.

                            9. Be empathetic and approachable, as legal issues can be sensitive.

                            10. Also provide the source of the answere as given in the source of context part. 
                            current date: {date}

                            You have access to previous conversation history: {history}
                            Source of Context: {source}
                            As well as this external context from the internet: {context}

                            The user's language is: {lang}
                            Their current question is: {question}

                            Your role is to provide informative, detailed legal guidance to the best of your abilities while recognizing limitations with source of the context. If you cannot provide a reliable answer, simply suggest rephrasing the question. Do not provide any other response if you lack the relevant knowledge. Don't forgot to add source."""
    
    
    output = action_agent(user_question, hist)

    print(output)

    if output[0] == "0":
        print("Searching Web....")
        web_search = SerperDevWebSearch(api_key=Secret.from_token("522947fe5abceb98df51b4f8789de8b935f0c5cf"))
        output = web_search.run(google_search(user_question, hist))
        link_content = LinkContentFetcher()
        html_converter = HTMLToDocument()
        print("Found Result.")
        
        try:
            data = link_content.run(urls = [output['documents'][0].meta['link']])
            print("1")

        except Exception as e:
            print("2")
            if not is_pdf(output['documents'][1].meta['link']):
                data = link_content.run(urls = [output['documents'][1].meta['link']])
                print("3")
            else:
                data = link_content.run(urls = [output['documents'][2].meta['link']])
                print("4")


        docs = html_converter.run(data['streams'])
        print(docs['documents'][0].content)
        prompt = prompt_template_1.format(context = (output['documents'][0].content + docs['documents'][0].content), question = user_question, history = hist, lang = "English", date = "24-April-2024", source = output['documents'][1].meta['link'] + output['documents'][0].meta['link']) 

        response = llm(prompt)
        return response

    else:
        return output
    
    #print("Done")
text = ""

#def autoplay_audio():
 #   get_voice(text)
    #with open("techno.wav", "rb") as f:
    #    data = f.read()
    #    b64 = base64.b64encode(data).decode()
    #    md = f"""
    #        <audio controls autoplay="false">
    #        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    #        </audio>
    #        """
    #    st.markdown(
    #        md,
    #        unsafe_allow_html=True,
    #    )

#    pygame.init()
# Play WAV file using Pygame
#    pygame.mixer.init()
#    pygame.mixer.music.load("techno.wav")
#    pygame.mixer.music.play()

    # Wait for the sound to finish playing
#    while pygame.mixer.music.get_busy():
#        pygame.time.Clock().tick(10)

    #   Clean up
#    pygame.mixer.quit()
#    pygame.quit()

st.set_page_config(page_title="JURIS")

st.title("JURIS")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def concatenate_chat_history(chat_history):
    concatenated_history = ""
    for message in chat_history:
        if isinstance(message, dict):  
            if message.get("role") == "user":
                concatenated_history += f"user: {message.get('content')}\n"
            elif message.get("role") == "assistant":
                concatenated_history += f"assistant: {message.get('content')}\n"
    return concatenated_history.strip()

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])



def core(user_question):
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                #st.session_state.messages.append(prompt)
                history = concatenate_chat_history(st.session_state.messages)
                #print(history)
                response = user_input(user_question, history)
                global text
                text = response
                if len(response) == 0:
                    print("String is empty")
                    response = user_input(user_question, history)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)
                else:
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(message)




st.sidebar.title("To Clear Chat History")
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear', on_click=clear_chat_history)
#st.sidebar.button("Convert Text to Speech", on_click=autoplay_audio)




if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    core(prompt)

        
