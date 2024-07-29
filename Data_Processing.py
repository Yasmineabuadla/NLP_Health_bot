import pandas as pd
import re
import json

df_train = pd.read_csv('./Data/MTS-Dialog-TrainingSet.csv')
df_val = pd.read_csv('./Data/MTS-Dialog-ValidationSet.csv')
df_test_chat = pd.read_csv('./Data/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')
df_test_sum = pd.read_csv('./Data/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv')



# Function to find unique characters
def find_unique_characters(text_series):
    unique_characters = set()
    for text in text_series:
        unique_characters.update(set(text))
    return unique_characters

# Find unique characters in the dialogue column
unique_characters = find_unique_characters(df_train['dialogue'])

# Convert to a sorted list for better readability
unique_characters = sorted(unique_characters)


# Updated data cleaning function to remove tabs, new lines, carriage returns, and replace "..." with "etc."
def clean_text(text):
    text = re.sub(r'[\t\n\r]', ' ', text)  # Remove tabs, new lines, and carriage returns
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'(\d+)-year-old', r'\1 year old', text)  # Replace "-" with space in "number-year-old"
    text = re.sub(r'\.\.\.', 'etc.', text)  # Replace "..." with "etc."
    text = text.strip()  # Remove leading and trailing spaces
    return text

# Apply the updated cleaning function
df_train['dialogue'] = df_train['dialogue'].apply(clean_text)
df_val['dialogue'] = df_val['dialogue'].apply(clean_text)
df_test_chat['dialogue'] = df_test_chat['dialogue'].apply(clean_text)
df_test_sum['dialogue'] = df_test_sum['dialogue'].apply(clean_text)


# Function to find unique parties involved in the dialogues
def find_unique_parties(text_series):
    unique_parties = set()
    for text in text_series:
        # Find all unique parties followed by a colon (e.g., 'Patient:', 'Doctor:', 'Guest_clinician:', 'Doctor_1:', 'Guest_family_1:')
        parties = re.findall(r'\b[\w]+:', text)
        unique_parties.update(parties)
    # Clean up the results by removing any anomalies
    clean_parties = {party for party in unique_parties if party.lower() not in {'following:'}}
    return clean_parties

# Find unique parties in the dialogue column using the updated function
unique_parties_train = find_unique_parties(df_train['dialogue'])
# Convert to a sorted list for better readability
unique_parties_train = sorted(unique_parties_train)

unique_parties_val = find_unique_parties(df_val['dialogue'])
# Convert to a sorted list for better readability
unique_parties_val = sorted(unique_parties_val)

unique_parties_test_chat = find_unique_parties(df_test_chat['dialogue'])
# Convert to a sorted list for better readability
unique_parties_test_chat = sorted(unique_parties_test_chat)


unique_parties_test_sum = find_unique_parties(df_test_sum['dialogue'])
# Convert to a sorted list for better readability
unique_parties_test_sum = sorted(unique_parties_test_sum)

# Define the mapping function with additional checks for misspellings
def map_to_doctor_or_patient(party):
    party_lower = party.lower()
    if any(term in party_lower for term in ['doctor', 'docotr', 'clinician', 'clinican']):
        return 'Doctor:'
    elif any(term in party_lower for term in ['patient', 'family']):
        return 'Patient:'
    else:
        return party

# Function to map parties in a dialogue
def normalize_dialogue(dialogue):
    parties = re.findall(r'\b[\w]+:', dialogue)
    for party in parties:
        normalized_party = map_to_doctor_or_patient(party)
        dialogue = dialogue.replace(party, normalized_party)
    return dialogue

# Apply the normalization to each dataframe's dialogue column
df_train['dialogue'] = df_train['dialogue'].apply(normalize_dialogue)
df_val['dialogue'] = df_val['dialogue'].apply(normalize_dialogue)
df_test_chat['dialogue'] = df_test_chat['dialogue'].apply(normalize_dialogue)
df_test_sum['dialogue'] = df_test_sum['dialogue'].apply(normalize_dialogue)


#############################################################################

# Base system prompt
BASE_SYSTEM_PROMPT = 'You are a medical chatbot. You should help the patient by answering their questions and providing advice about the following topic: {}'

# Function to create the dataset for an entire conversation
def create_conversation_dataset(section_header, messages):
    system_prompt = BASE_SYSTEM_PROMPT.format(section_header)
    return {
        "messages": [{"role": "system", "content": system_prompt}] + messages
    }
# Function to split the dialogue into patient questions and doctor answers
def split_dialogue(dialogue):
    split_dialogue = re.split(r'(Doctor:|Patient:)', dialogue)
    messages = []
    current_speaker = None
    for i in range(1, len(split_dialogue), 2):
        speaker = split_dialogue[i].strip()
        content = split_dialogue[i + 1].strip()
        if speaker == "Patient:":
            messages.append({"role": "user", "content": content})
            current_speaker = "Patient"
        elif speaker == "Doctor:":
            messages.append({"role": "assistant", "content": content})
            current_speaker = "Doctor"
    return messages

# Create the JSONL file
output_path = './train.jsonl'

with open(output_path, "w") as f:
    for convo_id in df_train['ID'].unique():
        convo_df = df_train[df_train['ID'] == convo_id]
        full_dialogue = " ".join(convo_df['dialogue'])
        section_header = convo_df['section_text'].iloc[0]  # Get the section header for the current conversation
        messages = split_dialogue(full_dialogue)
        conversation = create_conversation_dataset(section_header, messages)
        example_str = json.dumps(conversation)
        f.write(example_str + "\n")
output_path

# Create the JSONL file
output_path_val = './val.jsonl'
with open(output_path_val, "w") as f:
    for convo_id in df_val['ID'].unique():
        convo_df = df_val[df_val['ID'] == convo_id]
        full_dialogue = " ".join(convo_df['dialogue'])
        section_header = convo_df['section_text'].iloc[0]  # Get the section header for the current conversation
        messages = split_dialogue(full_dialogue)
        conversation = create_conversation_dataset(section_header, messages)
        example_str = json.dumps(conversation)
        f.write(example_str + "\n")
output_path_val

# Create the JSONL file
output_path_test_chat = './test_chat.jsonl'
with open(output_path_test_chat, "w") as f:
    for convo_id in df_test_chat['ID'].unique():
        convo_df = df_test_chat[df_test_chat['ID'] == convo_id]
        full_dialogue = " ".join(convo_df['dialogue'])
        section_header = convo_df['section_text'].iloc[0]  # Get the section header for the current conversation
        messages = split_dialogue(full_dialogue)
        conversation = create_conversation_dataset(section_header, messages)
        example_str = json.dumps(conversation)
        f.write(example_str + "\n")
output_path_test_chat

# Create the JSONL file
output_path_test_sum = './test_sum.jsonl'
with open(output_path_test_sum, "w") as f:
    for convo_id in df_test_sum['ID'].unique():
        convo_df = df_test_sum[df_test_sum['ID'] == convo_id]
        full_dialogue = " ".join(convo_df['dialogue'])
        section_header = convo_df['section_text'].iloc[0]  # Get the section header for the current conversation
        messages = split_dialogue(full_dialogue)
        conversation = create_conversation_dataset(section_header, messages)
        example_str = json.dumps(conversation)
        f.write(example_str + "\n")
output_path_test_sum