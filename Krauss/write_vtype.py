"""
write_vtype.py - Generate vType XML with SBI-sampled parameters
"""

def write_vtype_file(theta, output_path):
    """
    theta: [speedFactor, speedDev, sigma, tau] - only for type_3
    Other types get fixed default values
    """
    speedFactor, speedDev, sigma, tau = theta

    xml_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<routes>\n'
        '  <vType id="type_0" vClass="passenger"'
        ' length="5.0" accel="1.2" decel="2.0" emergencyDecel="6.0"'
        ' minGap="3.5" tau="1.8" actionStepLength="1.0"'
        ' maxSpeed="13.9" speedFactor="0.95" speedDev="0.1"'
        ' sigma="0.4" carFollowModel="Krauss" guiShape="passenger"/>\n'
        '  <vType id="type_1" vClass="passenger"'
        ' length="5.0" accel="2.0" decel="3.5" emergencyDecel="7.0"'
        ' minGap="3.0" tau="1.4" actionStepLength="1.0"'
        ' maxSpeed="15.0" speedFactor="1.0" speedDev="0.1"'
        ' sigma="0.3" carFollowModel="Krauss" guiShape="passenger"/>\n'
        '  <vType id="type_2" vClass="passenger"'
        ' length="5.0" accel="2.5" decel="4.0" emergencyDecel="8.0"'
        ' minGap="2.8" tau="1.3" actionStepLength="1.0"'
        ' maxSpeed="16.0" speedFactor="0.95" speedDev="0.1"'
        ' sigma="0.25" carFollowModel="Krauss" guiShape="passenger"/>\n'
        f'  <vType id="type_3" vClass="passenger"'
        f' tau="{tau:.4f}"'
        f' speedFactor="{speedFactor:.4f}"'
        f' speedDev="{speedDev:.4f}"'
        f' sigma="{sigma:.4f}"'
        f' carFollowModel="Krauss" guiShape="passenger"/>\n'
        '  <vType id="type_4" vClass="passenger"'
        ' length="5.0" accel="1.0" decel="1.8" emergencyDecel="6.5"'
        ' minGap="4.0" tau="2.0" actionStepLength="1.0"'
        ' maxSpeed="13.0" speedFactor="0.85" speedDev="0.1"'
        ' sigma="0.45" carFollowModel="Krauss" guiShape="passenger"/>\n'
        '</routes>\n'
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_content)