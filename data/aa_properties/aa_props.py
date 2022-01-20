from enum import Enum
import yaml
import os
from root_path import abspath_root


class AaProps(Enum):
    YAML_FILE = os.path.join(abspath_root, 'data', 'aa_properties', 'aa_props.yaml')

    with open(YAML_FILE, 'r') as stream:

        try:
            physicochemistry = yaml.load(stream=stream, Loader=yaml.FullLoader)
            LIST_ALL_20_AA = physicochemistry['list_all_20_aa']
            P_BETA = physicochemistry['Chou-Fasman_Beta-Strand_Preference']
            P_ALPHA = physicochemistry['Chou-Fasman_Alpha-Helix_Preference']
            P_T = physicochemistry['Chou-Fasman_Reverse_Turn_Preference']

        except yaml.YAMLError as exc:
            print(exc)


if __name__ == '__main__':
    # print(AaProps.LIST_ALL_20_AA.value)
    print(AaProps.P_BETA.value)
