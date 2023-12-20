# Ticket-To-Ride-Scorer

**Authors:**  John Soeder and Ian Matheson

## Introduction

Welcome to the Ticket to Ride Scoring System, a solution we created to enhance the gaming experience of one of our favorite board games, Ticket to Ride Europe. Say goodbye to tedious and error-prone manual scoring, and dive into a new era of efficient and accurate score calculation!

## What is Ticket to Ride?

Ticket to Ride, designed by Alan R. Moon, is a captivating train-themed board game where players strategically collect train cards to claim routes and earn points. Originally set in the US, various versions, such as Ticket to Ride: Europe, have added exciting twists to the gameplay.

The primary goal of Ticket to Ride is to accumulate the most points by completing destination tickets, connecting cities with train routes, and claiming the longest continuous route. An example final game state is shown below with key features labelled. We hope this image highlights the complexity of the scoring process and established the inefficiencies in the scoring process.

<p align="center">
<img width="650" alt="Screenshot 2023-12-19 at 10 12 08 PM" src="https://github.com/jsoeder20/Ticket-To-Ride-Scorer/assets/97808250/a5d096ce-4677-49dd-a858-219dfbcfb543">
</p>

## The Challenge

Scoring in Ticket to Ride can be time-consuming and prone to errors, especially with larger games. This scoring system aims to streamline the process using computer vision, reducing the scoring time and eliminating human errors.

## System Overview

1. **Detecting Trains:** Utilizing a specialized solution involving object detection models and labeled masks, the system identifies individual train spots on the board, classifying them as red, blue, yellow, black, green, or empty. Shown on the left, is a visual representation of the bounding boxes used to extract individual train spot images out of the cropped and resized image of final state of the game board. These bounding boxes were created using CVAT and each one is a associated with a label that will allow the scoring system to determine where each space is on the game board. On the right are the extract images showing individual spots from a completed game. These images are cropped, rotated, then cropped again to ensure that they all have the same orientation. The extracted images are classified using a simple CNN Trained on over 1000 train spot images. 

<p align="center">
<img width="300" alt="Screenshot 2023-12-19 at 10 13 12 PM" src="https://github.com/jsoeder20/Ticket-To-Ride-Scorer/assets/97808250/b4529280-985e-41b3-931a-f2ccf83d173c">

<img width="530" alt="Screenshot 2023-12-19 at 10 13 39 PM" src="https://github.com/jsoeder20/Ticket-To-Ride-Scorer/assets/97808250/4553ece6-2f2c-4aca-ad0a-13447ed7d94a">
</p>

2. **Detecting Train Stations:** Similar to the train detection process, the system identifies train stations on cities, classifying them by color. In this step, another simple CNN is used to classify the spots on the game boad where train stations would be placed. This model was trained on over 300 labelled city spot images.

<p align="center">
<img width="300" alt="Screenshot 2023-12-19 at 10 14 11 PM" src="https://github.com/jsoeder20/Ticket-To-Ride-Scorer/assets/97808250/1e2b545c-e5e1-406e-b60b-61fa2914d1aa">

<img width="530" alt="Screenshot 2023-12-19 at 10 14 29 PM" src="https://github.com/jsoeder20/Ticket-To-Ride-Scorer/assets/97808250/22a8a7ef-7474-48ac-b573-7be451b49c76">
</p>

4. **Scoring the Game:** Once the game state is determined, the system proceeds to score the game, including summing track values, adding route card bonuses, determining the longest consecutive track, and accounting for unused train stations.  

## How to Use

1. Play a game of Ticket to Ride Europe.
2. Take an image of the final game state. For best results, ensure trains are placed neatly and the board is well-lit (minimize glare if possible).
3. Apply a perspective crop on your board game image. This can be best done with Photoshop or any other photo editing software. Ensure that the cropped image includes the board only (see the example below).
4. Run the score game file and answer the following prompts.
5. Receive a detailed outline of your game score!

<p align="center">
<img width="650" alt="Screenshot 2023-12-19 at 10 18 29 PM" src="https://github.com/jsoeder20/Ticket-To-Ride-Scorer/assets/97808250/1a4de8a6-fa24-49de-ae4f-a4c311c1d3a7">
</p>

## Results and Benefits

The system delivers final scores and a detailed summary, reducing scoring time by half and minimizing player scoring errors. Even with manual perspective cropping, this method enhances the gaming experience.

## Future Considerations

Future considerations include automated perspective cropping, route card image processing, and enhanced rule enforcement.
