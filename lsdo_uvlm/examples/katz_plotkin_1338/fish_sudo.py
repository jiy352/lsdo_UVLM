
'''1.initalize fish simulation'''
fish_simulation = caddee(name='fish_simulation')

'''2. add fish geometry (streamline)'''
fish_simulation.add_geometry()
# endregion

'''3. (prescribed motions) add fish actuation (input: f, A, phis', outputs: mesh and velocities)'''
actuation_parameters = dict{
    'f':f,
    'A':A,
    'phi':phi,
}
fish_simulation.add_actuation(actuation_parameters)

'''4. add disciplinary model'''

fish_simulation.add_weight_estimation(weight_solver)
fish_simulation.add_hydrodynamics(UVLM_solver)
fish_simulation.add_battery_model(ecm_solver)
fish_simulation.add_energy_harvest_model(energy_harvest_solver)
fish_simulation.add_EoM(eom_solver)

'''5. specify objective and constraints'''
output_name = 'maximum_power'
output_name = 'final_SoC'
output_name = 'speed'
fish_simulation.add_outputs()

# finalize model
# add ODE parameters


'''6. optimization scripts'''
# add objective
# add constraints
