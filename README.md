# iiwa-wsg-gym
 
## Skill Learning using Policy Gradient Reinforcement Learning for Robotic Arm Manipulator ( Kuka IIWA 14 with Schunk WSG 50 gripper )

Long horizon tasks are very difficult to solve in a complex environment, it becomes more challenging if we have to perform dexterous manipulation
which involves non-pick&place tasks. We aim to divide a task into smaller subtasks using a TAMP, i.e a task and motion planner
(which is not involved in this part of the project), thus making the long horizon problem easier to solve. We employ RL based policies 
which are trained using policy gradient algorithms like PPO (Proximal Policy Optimisation), SAC (Soft Actor-Critic).

 We aim to create a skill repository which would contain multiple skills like `reach`, `push`, `pull`, `wipe`, etc which could be triggered by the TAMP.

 For example  consider this scene.
 
<img width="268" height="245" alt="1" src="https://github.com/user-attachments/assets/cb254664-4d0f-45bd-a088-932a3d2a81a3" />

If a user wants to `put the cube in the cabinet's top drawer`. Subtasks generated will be.
```
1. Reach top drawer handle.
2. Grasp top drawer handle.
3. Pull top drawer handle.
4. Release top drawer handle.
5. Reach cube.
6. Grasp cube.
7. Move cube to top drawer.
8. Release cube.
9. Reach top drawer handle.
10. Grasp top drawer handle.
11. Push top drawer handle.
12. Release top drawer handle.

---DONE---
```
------------------------------------------------------------------------------------------------------------------------------------

Skill Name : `reach`




https://github.com/user-attachments/assets/4084a5f7-e4e2-4ad4-ab6c-b7be6d5288f2


The `reach` skill is a basic skill which empowers the robot to reach a reachable position and valid SO(3) rotation in the workspace. 
------------------------------------------------------------------------------------------------------------------------------------

Skill Name : `push`


https://github.com/user-attachments/assets/8f84692b-c1fc-4337-89d6-56ad90af38be




