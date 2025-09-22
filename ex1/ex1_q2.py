# Moses the mouse
# state: position- (m, n)
# action: up/ right
# cost: sum of rewards- rt=1 if there is cheese in st o.w. 0

# horizon= n+m-2

# number of trajectories: c(n+m-2, n-1)


#  If both mice ignore each other’s existence and act’optimal’ with respect to the original problem, they will divide their detections.

# 4 actions, (n+m-2)**2 states

# For K mice:
# 2**K actions, (n+m-2)**K states
