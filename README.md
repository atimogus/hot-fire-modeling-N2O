modeling hot fire test of a self pressurized N2O rocket motor

in order to run this code you must have N2Omodel-izgaranje.py and functions.py in the same folder, open N2Omodel-izgaranje.py in your favourite IDE, tweak initial conditions and run your simulation
simulation will take about 1 minute to execute (depends on how much of oxidizer and fuel you have and mass flow rate) if high mass and low mass fuel rate longer you will wait because CEApy library is very slow at calculating 

![image](https://github.com/atimogus/hot-fire-modeling-N2O/assets/52748147/b93205de-29f7-473d-a70d-9cbcf365fcb4)

compared with HalfCatSim this python model has error of ~2% if accounted for input error, it can be said that error of this model is less than 2%

big thanks to r/rocketry community, 
https://github.com/rnickel1/HRAP_Source ; 
Sunride student team https://www.linkedin.com/company/team-sunride/?originalSubdomain=uk ; 
https://github.com/miamrljic03/NitrousOxide_Thermo_Diagrams ; 
https://www.halfcatrocketry.com/halfcatsim
