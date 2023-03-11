import re
import tensorflow as tf

def split_into_sentences(text):
    # Define a regular expression for matching sentence boundaries
    sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')

    # Use the regular expression to split the text into sentences
    sentences = sentence_pattern.split(text)

    # Remove any leading or trailing whitespace from each sentence
    sentences = [s.strip() for s in sentences]

    return sentences

# Make a preprocessing function
def split_chars(text):
    return " ".join(list(text))

def preprocess_text(text: str):
    sentences= split_into_sentences(text)
    sentences_char= [split_chars(sentence) for sentence in sentences]
    total= len(sentences)
    index= [num + 1 for num in range(total)]
    index_one_hot= tf.one_hot(index, depth= 15)
    total= [total for _ in range(total)]
    total_one_hot= tf.one_hot(total, depth= 20)
    data= tf.data.Dataset.from_tensor_slices((index_one_hot, total_one_hot, sentences, sentences_char)).batch(32).prefetch(tf.data.AUTOTUNE)
    labels= tf.data.Dataset.range(len(sentences))

    return tf.data.Dataset.zip((data, labels))

def process_output(data, sentences, class_names, model):
    outputs= model.predict(data) 
    preds= tf.squeeze(tf.argmax(outputs, axis= 1))
    return [{class_names[pred]:sentence} for pred, sentence in zip(preds, sentences)]

def output_to_text(model_output: list [dict], class_names: list [str]):
    text= {}
    for sentence in model_output:
        key= list(sentence.keys())[0]
        if text.get(key, None) != None:
            text[key] += f" {sentence.get(key)}"
        else:
            text[key] = sentence.get(key)

    return list(text.keys()), text