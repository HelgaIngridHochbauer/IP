# Room Transformation Studio

Change anything, from furniture to color-scheme of your house with this AI tool

## Introduction
Room Transformation Studio is an innovative application designed to assist users in redesigning and
reimagining their living spaces using AI-powered tools. This documentation provides a comprehensive
overview of the application, its functionality, and implementation.





## Features
<img width="1329" height="693" alt="image" src="https://github.com/user-attachments/assets/250456fd-e7df-4c23-97e9-1b63d78b8026" />

<img width="1315" height="658" alt="image" src="https://github.com/user-attachments/assets/c9570917-75cc-4ef0-b98a-22cc627a335d" />

• AI-powered room transformation using the Stable Diffusion model.

• Interactive style quiz to determine user preferences.

• Customizable design elements such as colors, furniture, and wood finishes.

• User-friendly interface built with Gradio.


The Stable Diffusion model is a state-of-the-art machine learning framework designed for generating
high-quality images from textual descriptions. 

The prompt generation feature is a critical component of the Room Transformation Studio. It ensures
that user inputs are effectively translated into coherent prompts for the Stable Diffusion model.

The application employs a dynamic prompt-building logic that integrates various user inputs, such as
room type, design elements, and aesthetic preferences. The resulting prompt is both descriptive and
specific, enabling accurate image generation. Below is a detailed explanation of the logic:

• Room Type: Specifies the type of room to be transformed (e.g., Bedroom, Kitchen).

• Change Options: Highlights elements the user wishes to modify (e.g., Colors, Furniture).

• Aesthetic: Defines the design style (e.g., Modern, Classic).

• Color and Wood Details: Incorporates color preferences and wood finish types for a personalized
touch

The Gradio interface is a key component of the Room Transformation Studio, providing an intuitive and user-friendly platform for interaction. Gradio simplifies the process of engaging with AI models by
offering a web-based interface that requires no technical expertise.


## How to use it
1. upload an image
2. add your preferences
3. the AI model will be prompted such that it will generate a new image with your new room

## Examples
<img width="1358" height="683" alt="image" src="https://github.com/user-attachments/assets/b5f08e05-b0fb-41ca-91ae-5f60f0dca612" />

