import gradio as gr
from theme_classifier import ThemeClassifier
import plotly.express as px
from character_network import CharacterNetworkGenerator, named_entity_recognizer
from text_classification import jutsu_classifier
import huggingface_hub
import os
from character_chatbot import character_chatbot
from dotenv import load_dotenv
load_dotenv()

def get_themes(theme_list, subtitles_path, save_path):
    theme_list = [theme.strip() for theme in theme_list.split(',')]

    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)
    output_df.columns = [col.strip() for col in output_df.columns]
    
    #Remove Dialogue from the Theme List
    theme_list_no_dialogue = [theme for theme in theme_list if theme.lower() != 'dialogue']
    output_df = output_df[theme_list_no_dialogue]
        
    output_df = output_df.sum().reset_index()
    output_df.columns = ['theme', 'score']
    
        # Visualizing the Output using Plotly
    fig = px.bar(
        output_df,
        x='score',
        y='theme',
        orientation='h',
        color='score',
        color_continuous_scale='viridis',
        title='Series Themes',
        labels={'score': 'Score', 'theme': 'Theme'},
        width=500,
        height=300 + (len(theme_list) * 2)
    )
    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="Theme",
                yaxis=dict(
            automargin=True,
            tickfont=dict(size=12),
        ),
        margin=dict(l=200, r=20, t=50, b=20)
    )

    return fig
    
#     #Visualizing the Output
#     output_chart = gr.BarPlot(
#     output_df,
#     x='theme',
#     y='score',
#     title='Series Themes',
#     tooltip=['theme', 'score'],
#     color='score',
#     vertical=False,
#     color_scale="viridis",
#     width=800, # Increased width
#     height=300, # Increased height
#     layout={'xaxis': {'tickangle': 45, 'tickfont': {'size': 10}}}
# )
    
    
#     # Get themes
#     themes_df = theme_classifier.get_themes(subtitles_path, save_path)
    
#     # # Plotting
#     # plot = gr.BarPlot()
#     # plot.update(themes_df)
    
#     # return plot
#     return output_chart

def get_character_network(subtitles_path, ner_path):
    ner = named_entity_recognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)
    
    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    
    return html

def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    # Load the model
    classifier = jutsu_classifier(
        model_path=text_classification_model,
        data_path=text_classification_data_path,
        huggingface_token=os.getenv('hugging_face_token')
    )
    
    # Classify the text
    output = classifier.classify_jutsu(text_to_classify)
    output = output[0]
    return output

def chat_with_character_chatbot(message, history):
    character_instance = character_chatbot("kartikeyadatta/Naruto_Meta-Llama-3-8B-Instruct",
                                         huggingface_token=os.getenv('hugging_face_token'))
    
    output = character_instance.chat(message, history)
    output = output['content'].strip()
    return output

def main():
    with gr.Blocks() as iface:
        
        #Theme Classification section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1 style='text-align: center;'>Theme Classification (Zero Shot Classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                         plot = gr.Plot(label="Theme Plot")
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes", placeholder="Enter themes separated by commas")
                        subtitltes_path = gr.Textbox(label="Subtitle or script Path", placeholder="Enter path to subtitles")
                        save_path = gr.Textbox(label="Save Path", placeholder="Enter path to save the results")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitltes_path, save_path], outputs=plot)
                        
                        
        #Charactr Network section              
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1 style='text-align: center;'>Character (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML(label="Character Network")
                    with gr.Column():
                        subtitltes_path = gr.Textbox(label="Subtitle or script Path", placeholder="Enter path to subtitles/Script")
                        ner_path = gr.Textbox(label="NERs Save Path", placeholder="Enter path to save the results")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitltes_path, ner_path], outputs=[network_html])
                        
                        
        #Text Classification with LLM's             
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1 style='text-align: center;'>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Text Classification Output", placeholder="Enter text to classify")
                    with gr.Column():
                        text_classification_model = gr.Textbox(label="Model path", placeholder="Enter models path")
                        text_classification_data_path = gr.Textbox(label="Data Path", placeholder="Enter path to retrieve the data from")
                        text_to_classify = gr.Textbox(label="Text inpout", placeholder="Pleae enter the text to classify")
                        classify_text_button = gr.Button("Classify Text (Jutsu)")
                        classify_text_button.click(classify_text, inputs=[text_classification_model, text_classification_data_path, text_to_classify], outputs=[text_classification_output])
    
        #Character Chatbot section
        with gr.Row():
             with gr.Column():
                gr.HTML("<h1 style='text-align: center;'>Character Chatbot 🍥</h1>")
                gr.ChatInterface(chat_with_character_chatbot)
                            
    iface.launch(share=True)
                    
    
if __name__ == "__main__":
    main()