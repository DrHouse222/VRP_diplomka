
Felipe Á, Ortuño MT, Righini G, Tirado G (2014) A heuristic approach for the green vehicle routing problem with multiple technologies and partial recharges. Transportation Res. Part E 71:111–128.

Instances for the electric vehicle routing problem with multiple recharge technologies
Number of instances: 41
------------------------------------------------------------------------------------------
<network>
- the node with type=0 is the depot
- the nodes with type=1 are the customers
- the nodes with type=2 are the charging stations (CSs)
- nodes with type=2 define the type of charging station in tag <has_tech>

<fleet>
- There is just one type of electric vehicle in the 41 instances
- <battery_capacity> defines the total energy capacity
- <function has_tech="X"> defines the charging function of the electric vehicle when charged with technology of type X
 - <rho> unit recharge speed
- <gamma> is the unit recharge cost

<requests>
- Each customer has 1 request
- Each node (including stations) has a fixed service time


