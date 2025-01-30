# Vaccum-Cleaner-Agent
The goal in this project is to practice design and implementation of the common search algorithms used in planning for intelligent agents in a fully visible environment. The environment is a cleaning robot which moved from room to room in a grid and clean the rooms/cells if dirty. Dirty cells are specified with grey color. There are wall cells which restrict the free movement, and the robot must find its ways around the wall blocks. Each move from a cell to its neighboring cell costs 1.

The robot finds its way around the grid by pre-planning its path to the next dirty room. For planning the robot has access to 3 uninformed (DFS, BFS, UCS) and 2 informed (Greedy, A*) search algorithms. There is also a reflex agent style planning in which robot moves around in a random move, if no neighboring dirty room exists, otherwise moves to the dirty room and clean it. 
For each path to the next dirty room, the explored and path cells are displayer with orange and pink colors. There are 2 counters at the top of the GUI that display the accumulated explored count and the total path count so far. Both these numbers are accumulative.

This assignment is provided in the form of a shell module, and you are going to fill in the specified missing parts. These parts currently have print messages saying: "For students to implement". Once you have done a part, remove the printed message. The GUI and mechanics of the application should work fine as it is, and the above functionalities work only after you implement the corresponding algorithms. 
 
The scripts in this project contain lots of comments and instruction. Read them carefully. They are part of the assignment description.

The way implementation works is that the path and explored list are computed when you select one of the search algorithms from the menu and click on run button.
 
The agent starts from the middle of the grid. The next search is performed automatically for the next dirty room and so on, till all grid is clean.


Cost:  
The basic cost function is the Step count, which is the number of step (moving from a cell to a neighboring cell). There is a cost dropdown menu to choose other costing options. Available options are: StepTurn, StayLeft, StayUp. The StepTurn charges extra for each turn using a formula described below. 
A turn of 90’ should costs 3 compare to cost of 1 for each step. For example for currently heading north state, a 90 turn clockwise to head east and then move to the right cell , the cost should be 3 + 1=4. This should result in the agent preferring the path with smaller number of turns. 
The StayLeft should be designed such that encourages the agent to clean the left half of the grid cells first before moving to the right half. You must come up with a simple cost function which embodies such tendency. Similarly, StayUp should cause the agent to first clean the rooms in top half of the grid. For both of the last 2 cost function we only are looking for the tendency and not matching any exact cost numbers. Both StayLeft and StayUp are used with UCS algorithm.

Heuristic:

For heuristic we have 2 choices: Manhattan and Euclid distance.
