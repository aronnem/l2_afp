------------------------------------------------------------
--- Note that the use of base_config.lua is entirely optional.
--- There is a small number of global variables that you
--- need to create, you can do this however you want.
---
--- The variables are:
--- logger - A LogImp
--- forward_model - A ForwardModel
--- solver - A ConnorSolver
--- initial_guess - A InitialGuess
--- number_pressure_level - Integer giving the number of pressure 
---    levels. This is used to size the output file, so it should
---    be the maximim pressure
--- number_aerosol - Integer giving number of Aerosol particles. 
---    Can be 0. This is used to size the output file.
--- number_band - Integer giving number of spectral bands.
--- iteration_output - Boolean. True if we should write out iteration
---   output
--- register_output - A VectorRegisterOutput giving the
---   list of output that should be generated. This list can empty if
---   no output is desired.
------------------------------------------------------------

require "oco_base_config"

config = OcoBaseConfig:new()

--config.fm.atmosphere.absorber.O2.absco = "v5.0_alpha/o2_v151005_cia_mlawer_v151005r1_narrow.hdf"
--config.fm.atmosphere.absorber.CO2.absco = "v5.0_alpha/co2_devi2015_wco2scale=nist_sco2scale=unity.hdf"
--config.fm.atmosphere.absorber.H2O.absco = "v5.0_temporary/h2o_hitran12.hdf"

--config.fm.atmosphere.absorber.O2.table_scale = 1.0
--config.fm.atmosphere.absorber.CO2.table_scale = { 1.0, 1.0, 1.004 }

config:do_config()


