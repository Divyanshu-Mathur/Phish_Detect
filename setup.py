from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    req_lst:List[str]=[]
    try:
        with open("requirements.txt","r") as file:
            lines = file.readlines()
            
            for line in lines:
                requirement = line.strip()
                if requirement and requirement !="-e .":
                    req_lst.append(requirement)

        return req_lst  
                    
    except FileNotFoundError :
        print("Requirements file not found")
        

setup(
    name="Network Security",
    version="0.0.0.1",
    author="Divyanshu Mathur",
    packages=find_packages(),
    install_requires = get_requirements()
)             
    