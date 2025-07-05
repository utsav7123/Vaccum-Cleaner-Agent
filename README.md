# Vacuum Planning Agents: Search and Pathfinding in Grid Environments

## Overview

This project implements and evaluates classical search algorithms for intelligent agent planning in a grid-based environment. The environment simulates a robot vacuum cleaner that must clean all dirty rooms while navigating around obstacles and walls.

> **Course:** CMPT 310 – Artificial Intelligence, Simon Fraser University (Winter 2025)

---

## Introduction

The goal of this project is to practice the design and implementation of common search algorithms used in planning for intelligent agents in a fully visible environment.  
The environment consists of a cleaning robot that moves from room to room in a grid and cleans the rooms/cells if dirty. Dirty cells are specified with grey color. There are wall cells which restrict free movement, and the robot must find its way around these obstacles. Each move from a cell to its neighboring cell costs 1.

The robot finds its way around the grid by pre-planning its path to the next dirty room. For planning, the robot has access to three uninformed (DFS, BFS, UCS) and two informed (Greedy, A*) search algorithms. There is also a reflex agent style, in which the robot moves randomly if no neighboring dirty room exists, otherwise moves to the dirty room and cleans it.

For each path to the next dirty room, the explored and path cells are displayed with orange and pink colors. There are two counters at the top of the GUI that display the accumulated explored count and the total path count so far. Both numbers are accumulative.

---

## Features

- **Agent and Environment Simulation**
    - 2D grid world with dirty cells and wall (obstacle) cells.
    - Robot agent plans and executes moves to clean all dirty rooms.
    - GUI visualizes agent, obstacles, dirty and clean cells.

- **Implemented Search Algorithms**
    - **Uninformed Search:**  
      - Breadth-First Search (BFS)
      - Depth-First Search (DFS)
      - Uniform-Cost Search (UCS)
    - **Informed Search:**  
      - Greedy Search (Manhattan/Euclidean heuristic)
      - A* Search (Manhattan/Euclidean heuristic)
    - **Reflex Agent:**  
      - Baseline agent acting on immediate percepts

- **Custom Cost Functions**
    - `Step`: Uniform cost per move.
    - `StepTurn`: Adds extra cost for each 90-degree turn (turn = 3, move = 1).
    - `StayLeft`: Encourages agent to clean the left half of the grid first.
    - `StayUp`: Encourages agent to clean the top half of the grid first.

- **Heuristics**
    - **Manhattan Distance**
    - **Euclidean Distance**

- **Visualization and GUI**
    - **Tkinter-based interface**  
      - Interactive grid with agent, walls, dirt, and clean cells
      - Path highlighted in orange; explored nodes in pink
      - Real-time counters for steps taken and nodes explored

- **Extensible, Modular Codebase**
    - Python classes for Agents, Environments, Problems, and Utilities
    - Easily adaptable for other search/planning scenarios

---

## Assignment Objectives

- Practice implementation of classic AI search algorithms.
- Compare uninformed vs. informed search strategies in agent navigation tasks.
- Explore the impact of different cost functions and heuristics on path quality.
- Experiment with environment configurations (walls, dirt placement, cost/heuristic options).

---

## Cost Functions (Details)

- **Step**: Each move to a neighboring cell has a cost of 1.
- **StepTurn**: Adds a cost of 3 for every 90-degree turn plus 1 for the move. The agent will prefer paths with fewer turns.
- **StayLeft**: Encourages the agent to prioritize cleaning the left half of the grid.
- **StayUp**: Encourages the agent to prioritize cleaning the upper half of the grid.
- **Heuristics**: For Greedy and A* searches, either Manhattan or Euclidean distance to the nearest dirty room can be selected.

---

## Usage

### Launching the GUI

Run the main script to start the graphical user interface:

    python vacuum_search.py

### Command-Line Options

You can run the project from the command line using: 
                 
    python vacuum_search.py -s searchAlgorithm -c costFunction -r heuristic

Where:

 -  -s specifies the search algorithm: BFS, DFS, UCS, Greedy, A*, or Reflex

 -  -c specifies the cost function: Step, StepTurn, StayLeft, StayUp

 -  -r specifies the heuristic: Manhattan, Euclid

Example:

     python vacuum_search.py -s A* -c StepTurn -r Manhattan

## Evaluation

   - GUI will display explored states and path cells for each algorithm run.

   - Two counters at the top track the total number of explored states and total path cost.

   - Assignment provides expected output ranges for each search/cost/heuristic combination.

   - For StayLeft and StayUp, agent cleaning order is observable on the GUI.

### Project Structure

    .
    ├── agents.py           # Agent, environment, and grid world base classes
    ├── search.py           # Search algorithms and problem definitions
    ├── utils.py            # Utility functions: distance, grid operations, etc.
    ├── vacuum_search.py    # Main script with GUI and environment setup
    ├── assignment.docx     # Assignment instructions and rubric
    ├── [other files]

## Technologies Used

   - Python 3

   - Tkinter (GUI)

   - NumPy (for distance metrics and math utilities)

## Credits

This project is adapted from the CMPT 310 (Artificial Intelligence) assignment at Simon Fraser University, Winter 2025.
Code structure and assignment specification provided by course instructors.
## License

For educational use only. Do not redistribute without permission. If reusing or extending for other coursework, please credit the original source.



