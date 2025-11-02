# Lesson 3 Set Up
## Step 1: Install Visual Studio Code and Git
- https://git-scm.com/install/
- https://code.visualstudio.com/download
## Step 2: Clone the repo
Go to your powershell/terminal. Enter
```bash
git clone https://github.com/extremefattypunch/Project-Svarog-Orbital-Public-Lesson-2025/
```
## Step 3
Go to visual studio code. Download the Julia Extension

<img width="300" height="270" alt="image" src="https://github.com/user-attachments/assets/adbb5181-0400-40bc-976d-9f127ed06924" />

Next find and open the folder "Project-Svarog-Orbital-Public-Lesson-2025" which you just cloned in the location you ran the pervious command

Then run the file "earth_orbit_acc.jl"
<img width="725" height="359" alt="image" src="https://github.com/user-attachments/assets/7d103d3c-f734-4d60-8de7-a24ea8faead0" />

You will run into some errors since you have not installed the required modules. DO NOT CLOSE THE REPL WINDOW.
## Step 4
Instead paste the following(`ctrl-shift-v`) into the terminal
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

<img width="623" height="378" alt="image" src="https://github.com/user-attachments/assets/c9c47ea1-82fb-432e-9fc6-9c423cddd166" />


like so. enter to run and the repl will start installing the packages.
## Step 4
Once done go back to the run repl button at the top the visual studio code window like in step 3 and press it to run.
<img width="1608" height="1074" alt="image" src="https://github.com/user-attachments/assets/e17ead37-5f3e-4c36-bc5a-00da4861eac6" />

when successful you will see plots created in a side pane on the visual studio code window as seen here. Click on it and press the left right arrows to navigate around the plots.
