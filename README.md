## My Custom AI Gaming Project

## Introduciton

Welcome to my class project, a testament to my journey in artificial intelligence and game theory. This project showcases my implementation of a sophisticated AI agent designed to play a strategic game. The core of my solution is built upon the Monte Carlo Tree Search (MCTS) algorithm, enhanced with the Upper Confidence Bound (UCB) technique. This combination ensures that my AI agent not only plays effectively but also adapts to new strategies and challenges.

## Monte Carlo Tree Search (MCTS) with Upper Confidence Bound

At the heart of my AI player lies the Monte Carlo Tree Search (MCTS) algorithm, a powerful method for making optimal decisions in game theory. My implementation utilizes the Upper Confidence Bound (UCB) approach to balance exploration and exploitation. This allows my agent to explore various game states, learning from each encounter, and improving its decision-making process over time. The result is an AI player that is not just playing the game, but learning and evolving with each move.

## Testing Procedures

Final Evaluation Test
Before you submit, it's crucial to ensure the performance of the AI player. For an exemplary grade, the AI (referred to as student_agent) should consistently outperform the random_agent. Here's how to run this crucial test:
    ```
    python simulator.py --player_1 random_agent --player_2 student_agent --display 
    ```


## Autoplay Functionality Check
Verify the seamless interaction between your AI agent (student_agent) and the random_agent in autoplay mode. This step confirms the smooth operation of the game simulation
    ```
    python simulator.py --player_1 random_agent --player_2 student_agent --autoplay
    ```
