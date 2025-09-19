# SSA DebriSolver Completion 
## Idea: 
We aim to address space debris challenges using a Digital Twin, a virtual 
model that precisely simulates objects in Low Earth Orbit with real-time data 
from tracking networks (e.g., Space Surveillance Network) and sensors (e.g., 
LiDAR). Our focus includes: accurately tracking small debris (1 mm-10 cm) 
using machine learning algorithms to mitigate collision risks, predicting debris 
trajectories and optimizing avoidance maneuvers through dynamic 
simulations with tools like Systems Tool Kit (STK), and developing cost
effective removal strategies, such as drag sails, to prioritize high-risk debris, 
thereby enhancing space sustainability in alignment with the Saudi Space 
Agency’s objectives. 
## Approach: 
Our approach is creating a 2D plot using Matplotlib to represent a simplified 
resampling of Low Earth Orbit (LEO), which will serve as our Digital Twin 
environment. Within this space, we will create a vector representing a 
Debris Object (DB) in LEO. After building the 2D environment (Space) and 
initializing the debris vector, we will apply a machine learning algorithm 
(starting with Linear Regression) to predict where the vector will travel, with 
the goal of avoiding potential collisions. 
To visualize this, we will draw a circle around the predicted position to 
indicate the debris’ whereabouts and provide an Estimated Time of Arrival 
(ETA). Additional elements, such as a second vector to represent an active 
satellite, can be added to simulate collision-risk scenarios more realistically. 
The choice of data, ML library, and algorithm will be determined later 
through testing to evaluate which approach is most suitable and efficient for 
our project. 
