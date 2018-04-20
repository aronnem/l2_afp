from setuptools import setup

setup(name = "l2_afp",
      version = "0.0.1",
      author = "Aronne Merrelli",
      author_email = "aronne.merrelli@gmail.com",
      url = "https://github.com/aronnem/l2_afp", 
      classifers = [
        "Development Status :: 1 - Planning", 
        "Intended Audience :: Science/Research",
        "Programming Language :: Python"],
      packages = ["l2_afp", "l2_afp.utils"],
      package_data = {
        "l2_afp" : ["lua_configs/*.lua",]},
      install_requires=["numpy", "scipy", "h5py"],
      )
   
