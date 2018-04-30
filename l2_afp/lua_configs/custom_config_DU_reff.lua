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

config.fm.atmosphere.absorber.O2.absco = "v5.0.0/o2_v151005_cia_mlawer_v151005r1_narrow.hdf"
config.fm.atmosphere.absorber.CO2.absco = "v5.0.0/co2_devi2015_wco2scale=nist_sco2scale=unity.hdf"
config.fm.atmosphere.absorber.H2O.absco = "v5.0.0/h2o_hitran12.hdf"

--config.fm.atmosphere.absorber.O2.table_scale = 1.0
--config.fm.atmosphere.absorber.CO2.table_scale = { 1.0, 1.0, 1.004 }

-- turn on per-iteration output
config.iteration_output = true

-- turn on jacobian output
config.write_jacobian = true

-- force DU to be used as the only aerosol, along with the 3 scatterer 
-- types that are present in all soundings.
require "helper_functions"

-- Turn off Merra logic
config.fm.atmosphere.aerosol.creator = ConfigCommon.aerosol_creator

-- List aerosols to use
config.fm.atmosphere.aerosol.aerosols = {"SO", "DU", "ST", "Ice"}

-- Give information about the aerosols not already in the base configuration
config.fm.atmosphere.aerosol.SO = {
  creator = ConfigCommon.aerosol_log_shape_gaussian,
  apriori = function(self)
    --prior optical depth prior, height, width
    return ConfigCommon.lua_to_blitz_double_1d({-4.382026635,0.9,0.05})
  end,
  covariance = function(self)
    --uncertainties (prior covariance matrix)
    return ConfigCommon.lua_to_blitz_double_2d({{4.0,0,0},{0,0.04,0},{0,0,1e-4}})
  end,
  property = ConfigCommon.hdf_aerosol_property("SO"),
}

config.fm.atmosphere.aerosol.DU = {
  creator = ConfigCommon.aerosol_log_shape_gaussian,
  apriori = function(self)
    --optical depth prior, height, width
    return ConfigCommon.lua_to_blitz_double_1d({-4.382026635,0.9,0.05})
  end,
  covariance = function(self)
    return ConfigCommon.lua_to_blitz_double_2d({{4.0,0,0},{0,0.04,0},{0,0,1e-4}})
  end,
  property = ConfigCommon.hdf_aerosol_property("DU"),
}

config.fm.atmosphere.aerosol.ST = {
  creator = ConfigCommon.aerosol_log_shape_gaussian,
  apriori = function(self)
    return ConfigCommon.lua_to_blitz_double_1d({-5.11599580975408205124,0.03,0.05})
  end,
  covariance = function(self)
    return ConfigCommon.lua_to_blitz_double_2d({{3.24,0,0},{0,0.04,0},{0,0,1e-4}})
  end,
  property = ConfigCommon.hdf_aerosol_property("strat"),
}

config.fm.atmosphere.aerosol.Ice = {
  creator = ConfigCommon.aerosol_log_shape_gaussian,
  apriori = function(self)
    return ConfigCommon.lua_to_blitz_double_1d({-5.11599580975408205124,0.03,0.05})
  end,
  covariance = function(self)
    return ConfigCommon.lua_to_blitz_double_2d({{3.24,0,0},{0,0.04,0},{0,0,1e-4}})
  end,
  property = ConfigCommon.hdf_aerosol_property("ice_cloud_MODIS6_deltaM_1000"), --the ice cloud type used in v8.1
}


-- switch water cloud property table to the one that will be modified on the fly
-- assumes it is in the current working dir.
config.static_aerosol_file = "l2_aerosol_DU_variable.h5"
config.fm.atmosphere.aerosol.DU.property = ConfigCommon.hdf_aerosol_property("DU_variable")

config:do_config()
