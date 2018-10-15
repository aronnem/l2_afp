-- Base config for iris, using operational default B8

require "oco_base_config"

config = OcoBaseConfig:new()

config.fm.atmosphere.absorber.O2.absco = "v5.0.0/o2_v151005_cia_mlawer_v151005r1_narrow.hdf"
config.fm.atmosphere.absorber.CO2.absco = "v5.0.0/co2_devi2015_wco2scale=nist_sco2scale=unity.hdf"
config.fm.atmosphere.absorber.H2O.absco = "v5.0.0/h2o_hitran12.hdf"

-- turn on per-iteration output
config.iteration_output = true

-- turn on jacobian output
config.write_jacobian = true

-- Turn off Merra logic
config.fm.atmosphere.aerosol.creator = ConfigCommon.aerosol_creator

-- List aerosols to use
config.fm.atmosphere.aerosol.aerosols = {"Water1", "Water2"}

-- Note here we point to two different water cloud r_effs (8 and 9 um)
config.fm.atmosphere.aerosol.Water1 = {
  creator = ConfigCommon.aerosol_log_shape_gaussian,
  apriori = function(self)
    return ConfigCommon.lua_to_blitz_double_1d({-4.382,0.75,0.2})
  end,
  covariance = function(self)
    return ConfigCommon.lua_to_blitz_double_2d({{3.24,0,0},{0,0.04,0},{0,0,1e-4}})
  end,
  property = ConfigCommon.hdf_aerosol_property("wc_008"),
}

config.fm.atmosphere.aerosol.Water2 = {
  creator = ConfigCommon.aerosol_log_shape_gaussian,
  apriori = function(self)
    return ConfigCommon.lua_to_blitz_double_1d({-4.382,0.75,0.2})
  end,
  covariance = function(self)
    return ConfigCommon.lua_to_blitz_double_2d({{3.24,0,0},{0,0.04,0},{0,0,1e-4}})
  end,
  property = ConfigCommon.hdf_aerosol_property("wc_009"),
}


config:do_config()
