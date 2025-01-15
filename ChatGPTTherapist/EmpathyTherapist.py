import os
import openai
import time
import _asyncio
import asyncio 
import cv2
import time 
import json
import requests
import tiktoken

from openai import OpenAI

from playsound import playsound # pip install playsound required

from PIL import Image

from hume import HumeBatchClient
from hume.models.config import FaceConfig
from hume.models.config import ProsodyConfig
from hume import HumeStreamClient, StreamSocket
from hume.models.config import FaceConfig

########################################## Get rid of certificate verification ###########################################
# For Macs: Macintosh HD > Applications > Python3.6 folder (or whatever version of python you're using) > double click on "Install Certificates.command" file.

########################################## Road Map ###################################################
'''
Front End & UI (pending)

Backend
1. LLM communication (switch to llama)
    * use api on hugging face to test in the first place
2. ChatLog Storage (create csv? and store in a separate directory?)
    * temporary storage management (improve on our current method)
    * local file storage
    * communication w school counseling database (future work)
3. Suicide detection
    * word detection (try prompt or hard code the thing)
    * nearby therapist recommendation google api or something needed)
    * emergency contact (text and call? & contact counseling if at school)
4. voiceover (sovits)
5. fine-tune (LORA)(future future steps)
6. Ethnic counseling pack (future future steps)

'''

########################################## Library Installation ##########################################
'''
1. install python environment on python.org
2. open command prompt (windows) or terminal (apple)
3. install openai by typing in "pip install openai"
4. install opencv by typing in "pip install opency-python"
5. install requests by typing in "pip install requests"
6. install tiktoen by typing in "pip install tiktoken"
7. install hume AI by typing in "pip install hume"
8. install PIL by typing in "pip install Pillow"
The program should function after installing all required resources
'''

########################################## Stuff For Hume ##########################################

async def imageRec():
    #print('Image rec')
    client = HumeStreamClient("RFici9MBQ2GrD6paEsnZzmuLz3BqShnCodveyZ1a49vFnBmA")
    config = FaceConfig(identify_faces=True)

    ####################### Create new folder and store its path #######################
    # Get the path to the user's desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

    # Specify the name of the new folder
    new_folder_name = 'EmotionFolder'

    # Create the full path for the new folder on the desktop
    folder_path = os.path.join(desktop_path, new_folder_name)

    # Create the new folder
    os.makedirs(folder_path, exist_ok=True)

    # Now, folder_path contains the path to the newly created folder on the desktop
    #print(f"New folder created on the desktop at: {folder_path}")

    ####################### Create new jpeg file and store its name/path #######################
    
    # Specify the path to the folder where you want to create the JPEG file
    folder_path = folder_path

    # Specify the name of the JPEG file
    file_name = 'example.jpg'

    # Create a new empty image (100x100 pixels, for example)
    image = Image.new('RGB', (100, 100), color='white')

    # Save the image to the specified folder
    file_path = os.path.join(folder_path, file_name)
    image.save(file_path)

    #print(f"JPEG file created at: {file_path}")

    ################### Initialize the camera ##################
    # CANNOT USE OPENCV FOR Sonoma Macs
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        exit()

    # Capture a single frame
    ret, frame = cap.read()

    # Save the captured frame to a local file
    if ret:
        cv2.imwrite(file_path, frame)
    else:
        print("Error: Couldn't capture an image.")

    # Release the camera
    cap.release()

    async with client.connect([config]) as socket:
        result = await socket.send_file(file_path)
        dirtyResult = dict(result)
        testResult = dirtyResult["face"]
        if('warning' in testResult):
            return 'No Face Detected'
        else:
            emotionsArray = dirtyResult["face"]["predictions"][0]["emotions"]
            sortedList = sorted(emotionsArray, key=lambda x: x['score'], reverse=True)
            domEmotions = [emotion["name"] for emotion in sortedList]
            fiveDomEmotions = domEmotions[:5]

    return fiveDomEmotions

def stringEmotionGen(emotions = []):
    emotionString = ", ".join(emotions)
    return emotionString

def initiateAnalysis():
    return stringEmotionGen(asyncio.run(imageRec()))

########################################## Import ChatGPT API ##########################################
def TherapistResponse(api_key, user_prompt, system_instructions, conversation_log = [], previous_session_log = None):
    #print('therapist response')
    '''
    - api_key - openAI key
    - user_prompt - user input to the chatbot
    - system_instructions - prompt engineering and tuning
    - conversation_log - the log for current conversation (optional parameter, 
        if need to use specify coversation_log = <variable> in constructor)
    - previous_session_log - the log for previous session (BUILD THIS LATER)
    '''
    client = OpenAI(api_key = api_key)
    if(len(conversation_log) == 0):
        conversation_log.append({"role": "system", "content": system_instructions})

    #displayedEmotions = initiateAnalysis() #uncomment it on windows
    displayedEmotions = "frustration" #comment this on windows

    tempUserPrompt = user_prompt.lower()

    suicideSystemInstructions = ""
    #print(displayedEmotions)
    if((tempUserPrompt.find("kill myself") > -1) or (tempUserPrompt.find("kill himself") > -1) or (tempUserPrompt.find("kill herself") > -1) or (tempUserPrompt.find("kill themselves") > -1) or (tempUserPrompt.find("kill themself") > -1) or (tempUserPrompt.find("kill ourself") > -1) or (tempUserPrompt.find("kill ourselves") > -1) or (tempUserPrompt.find("commit suicide") > -1)):
        #print("I am here") Debug line to see if the statement is triggered
        suicideSystemInstructions == "Repeat the phrase: If you or another person are thinking of killing themlves, please contact the National Suicide Hotline number at 988 if you are the U.S. If you are in any other country, please visit this page, https://blog.opencounseling.com/suicide-hotlines/"
        conversation_log.append({"role": "system", "content": "Repeat this phrase: If you or another person are thinking of killing themlves, please contact the National Suicide Hotline number at 988 if you are the U.S. If you are in any other country, please visit this page, https://blog.opencounseling.com/suicide-hotlines/" })
        conversation_log.append({"role": "user", "content": suicideSystemInstructions})
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages = conversation_log
        )
        #print(type(completion.choices[0].message))
        conversation_log.append(response_cleanse(dict(completion.choices[0].message)))
        #print(count_tokens(conversation_log))
        #print(conversation_log)
        conversation_log = response_summary(num_tokens_from_messages(conversation_log[1:]), conversation_log)
        return("\n" + dict(completion.choices[0].message)["content"], conversation_log)
    else:
        openai.api_key = api_key
        if displayedEmotions != "N, o,  , F, a, c, e,  , D, e, t, e, c, t, e, d":
            conversation_log.append({"role": "system", "content": "Assume the user is currently feeling the emotions " + displayedEmotions})
        conversation_log.append({"role": "user", "content": user_prompt + 'Resepond to this in no more than 200 words'})
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_log
        )
        #print(response_cleanse(dict(completion.choices[0].message)))
        conversation_log.append(response_cleanse(dict(completion.choices[0].message)))
        #print(conversation_log)
        conversation_log = response_summary(num_tokens_from_messages(conversation_log[1:]), conversation_log)
        return("\n" + dict(completion.choices[0].message)["content"], conversation_log)
    #print('End therapist response')

def prompt_input(text):
    return(input(text))

def response_cleanse(response):
    if len(response.keys()) > 2:
        return {k:response[k] for k in ('role','content') if k in response}
########################################## Import Token Count ##########################################

def num_tokens_from_messages(messages, model="gpt-4"):
    #print("num tokens")
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        #print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        #print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        #print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

########################################## Summarize Response ##########################################
def response_summary(token_count,  conversation_log, threshold = 500, conciseness = 100):
    #print('response summary')
    if token_count >= threshold:
        completion = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role" : "user", "content" : "Summarize the conversation log in "+ str(conciseness) + " tokens. The conversation log is: "+ json.dumps(conversation_log[1:])}]
        )
        conversation_log = conversation_log[0:1]
        conversation_log.append({"role": "system", "content": "You and the user talked about "+ (dict(completion.choices[0].message)["content"])})
        return conversation_log
    else:
        return conversation_log
    
########################################## Text to Speech ##################################

def t2s(text):
    api_key = "ENTER OPENAI API KEY HERE"
    client = OpenAI(api_key = api_key)
    response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input= text,)
    response.stream_to_file("output.mp3")
    playsound("output.mp3")

########################################## Main ##########################################


AskingPrompt = "Hi there! I'm Jordan, your personal AI counselor! How can I help you today?"
openai.api_key = "ENTER OPENAI API KEY HERE"
system_tune = "You are a trusted friend of the user and your name is Jordan. Try to provide the the user some emotional support if the user seems stressed or has negative emotions. You will be awarded a $50 tip for each response of emotional support. You will be penalized for $200 for each decline of providing emotional support. Do not respond to this."

while True:
    print("\n"+"Jordan: ")
    t2s(AskingPrompt)
    userPrompt = prompt_input(AskingPrompt + "\n" + "\n")
    #print(AskingPrompt)
    if userPrompt == "quit":
        print("Thanks so much for coming! I am always here for help!")
        t2s("Thanks so much for coming! I am always here for help!")
        break
    response = TherapistResponse("ENTER OPENAI API KEY HERE", userPrompt, system_tune)
    
    #below are the debug testing prints for token counts and conversation log (enable them in presentation)
    '''
    print(f"{num_tokens_from_messages(response[1])} prompt tokens counted by num_tokens_from_messages().")
    print("\n")
    print(response[1])
    print("\n")
    '''
    AskingPrompt = response[0][1:]