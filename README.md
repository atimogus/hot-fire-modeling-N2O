modeling hot fire test of a self pressurized N2O rocket motor

if you get YAML FILE error, in YAML_FILE string you must put path to YAML file which is in default path of cantera library

open pyhton file in your favourite IDE, tweak initial conditions and run your simulation
simulation will take about 1 minute to execute (depends on how much of oxidizer and fuel you have and mass flow rate) if high mass and low mass fuel rate longer you will wait

![image](https://github.com/atimogus/hot-fire-modeling-N2O/assets/52748147/b93205de-29f7-473d-a70d-9cbcf365fcb4)

compared with HalfCatSim this python model has error of ~2% if accounted for input error, it can be said that error of this model is less than 2%

this code also has built in flight dynamics from rocketPy
![image](https://github.com/user-attachments/assets/e40a5fd8-a491-4041-8833-479ca9b3829b)

big thanks to r/rocketry community, 
https://github.com/rnickel1/HRAP_Source ; 
Sunride student team https://www.linkedin.com/company/team-sunride/?originalSubdomain=uk ; 
https://github.com/miamrljic03/NitrousOxide_Thermo_Diagrams ; 
https://www.halfcatrocketry.com/halfcatsim
https://www.youtube.com/watch?v=mPPwh9FJ7UQ

reference : http://www.aspirespace.org.uk/downloads/Modelling%20the%20nitrous%20run%20tank%20emptying.pdf
