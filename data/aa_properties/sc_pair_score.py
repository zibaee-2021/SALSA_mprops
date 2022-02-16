from collections import namedtuple
from src.utils import normalise
from data.aa_properties import read_props
hyd = read_props.read_hydrophobicity_scale(read_props.Scales.KD.value)

# hyd = normalise.norm_dict(hyd)

sc_comp = namedtuple('sc_comp', 'fit, Hbond, electrostat, hydroph, disulphide, pi_stack')
aa = 'A'
AA = sc_comp(fit=1.0, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd[aa]), disulphide=0.0, pi_stack=0.0)
AC = sc_comp(fit=0.5, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['C']), disulphide=0.0, pi_stack=0.0)
AD = sc_comp(fit=0.4, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['D']), disulphide=0.0, pi_stack=0.0)
AE = sc_comp(fit=0.4, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['E']), disulphide=0.0, pi_stack=0.0)
AF = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['F']), disulphide=0.0, pi_stack=0.0)
AG = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['G']), disulphide=0.0, pi_stack=0.0)
AH = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['H']), disulphide=0.0, pi_stack=0.0)
AI = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['I']), disulphide=0.0, pi_stack=0.0)
AK = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['K']), disulphide=0.0, pi_stack=0.0)
AL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['L']), disulphide=0.0, pi_stack=0.0)
AM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['M']), disulphide=0.0, pi_stack=0.0)
AN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['N']), disulphide=0.0, pi_stack=0.0)
AP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['P']), disulphide=0.0, pi_stack=0.0)
AQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Q']), disulphide=0.0, pi_stack=0.0)
AR = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['R']), disulphide=0.0, pi_stack=0.0)
AS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['S']), disulphide=0.0, pi_stack=0.0)
AT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['T']), disulphide=0.0, pi_stack=0.0)
AV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['V']), disulphide=0.0, pi_stack=0.0)
AW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['W']), disulphide=0.0, pi_stack=0.0)
AY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Y']), disulphide=0.0, pi_stack=0.0)

aa = 'C'
CA = AC
CC = sc_comp(fit=0.8, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd[aa]), disulphide=1.0, pi_stack=0.0)
CD = sc_comp(fit=0.4, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['D']), disulphide=0.0, pi_stack=0.0)
CE = sc_comp(fit=0.4, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['E']), disulphide=0.0, pi_stack=0.0)
CF = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['F']), disulphide=0.0, pi_stack=0.0)
CG = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['G']), disulphide=0.0, pi_stack=0.0)
CH = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['H']), disulphide=0.0, pi_stack=0.0)
CI = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['I']), disulphide=0.0, pi_stack=0.0)
CK = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['K']), disulphide=0.0, pi_stack=0.0)
CL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['L']), disulphide=0.0, pi_stack=0.0)
CM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['M']), disulphide=0.0, pi_stack=0.0)
CN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['N']), disulphide=0.0, pi_stack=0.0)
CP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['P']), disulphide=0.0, pi_stack=0.0)
CQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Q']), disulphide=0.0, pi_stack=0.0)
CR = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['R']), disulphide=0.0, pi_stack=0.0)
CS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['S']), disulphide=0.0, pi_stack=0.0)
CT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['T']), disulphide=0.0, pi_stack=0.0)
CV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['V']), disulphide=0.0, pi_stack=0.0)
CW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['W']), disulphide=0.0, pi_stack=0.0)
CY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Y']), disulphide=0.0, pi_stack=0.0)

aa = 'D'
DA = AD
DC = CD
DD = sc_comp(fit=0.8, Hbond=0.1, electrostat=-1.0, hydroph=(hyd[aa]+hyd[aa]), disulphide=0.0, pi_stack=0.0)
DE = sc_comp(fit=0.4, Hbond=0.1, electrostat=-1.0, hydroph=(hyd[aa]+hyd['E']), disulphide=0.0, pi_stack=0.0)
DF = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['F']), disulphide=0.0, pi_stack=0.0)
DG = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['G']), disulphide=0.0, pi_stack=0.0)
DH = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['H']), disulphide=0.0, pi_stack=0.0)
DI = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['I']), disulphide=0.0, pi_stack=0.0)
DK = sc_comp(fit=0.3, Hbond=0.1, electrostat=1.0, hydroph=(hyd[aa]+hyd['K']), disulphide=0.0, pi_stack=0.0)
DL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['L']), disulphide=0.0, pi_stack=0.0)
DM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['M']), disulphide=0.0, pi_stack=0.0)
DN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['N']), disulphide=0.0, pi_stack=0.0)
DP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['P']), disulphide=0.0, pi_stack=0.0)
DQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Q']), disulphide=0.0, pi_stack=0.0)
DR = sc_comp(fit=0.3, Hbond=0.1, electrostat=1.0, hydroph=(hyd[aa]+hyd['R']), disulphide=0.0, pi_stack=0.0)
DS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['S']), disulphide=0.0, pi_stack=0.0)
DT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['T']), disulphide=0.0, pi_stack=0.0)
DV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['V']), disulphide=0.0, pi_stack=0.0)
DW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['W']), disulphide=0.0, pi_stack=0.0)
DY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Y']), disulphide=0.0, pi_stack=0.0)

aa = 'E'
EA = AE
EC = CE
ED = DE
EE = sc_comp(fit=0.8, Hbond=0.1, electrostat=-1.0, hydroph=(hyd[aa]+hyd[aa]), disulphide=0.0, pi_stack=0.0)
EF = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['F']), disulphide=0.0, pi_stack=0.0)
EG = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['G']), disulphide=0.0, pi_stack=0.0)
EH = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['H']), disulphide=0.0, pi_stack=0.0)
EI = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['I']), disulphide=0.0, pi_stack=0.0)
EK = sc_comp(fit=0.3, Hbond=0.1, electrostat=1.0, hydroph=(hyd[aa]+hyd['K']), disulphide=0.0, pi_stack=0.0)
EL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['L']), disulphide=0.0, pi_stack=0.0)
EM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['M']), disulphide=0.0, pi_stack=0.0)
EN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['N']), disulphide=0.0, pi_stack=0.0)
EP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['P']), disulphide=0.0, pi_stack=0.0)
EQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Q']), disulphide=0.0, pi_stack=0.0)
ER = sc_comp(fit=0.3, Hbond=0.1, electrostat=1.0, hydroph=(hyd[aa]+hyd['R']), disulphide=0.0, pi_stack=0.0)
ES = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['S']), disulphide=0.0, pi_stack=0.0)
ET = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['T']), disulphide=0.0, pi_stack=0.0)
EV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['V']), disulphide=0.0, pi_stack=0.0)
EW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['W']), disulphide=0.0, pi_stack=0.0)
EY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Y']), disulphide=0.0, pi_stack=0.0)

aa = 'F'
FA = AF
FC = CF
FD = DF
FE = EF
FF = sc_comp(fit=0.8, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd[aa]), disulphide=0.0, pi_stack=1.0)
FG = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['G']), disulphide=0.0, pi_stack=0.0)
FH = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['H']), disulphide=0.0, pi_stack=0.0)
FI = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['I']), disulphide=0.0, pi_stack=0.0)
FK = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['K']), disulphide=0.0, pi_stack=0.5)
FL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['L']), disulphide=0.0, pi_stack=0.0)
FM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['M']), disulphide=0.0, pi_stack=0.0)
FN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['N']), disulphide=0.0, pi_stack=0.0)
FP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['P']), disulphide=0.0, pi_stack=0.0)
FQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Q']), disulphide=0.0, pi_stack=0.0)
FR = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['R']), disulphide=0.0, pi_stack=0.5)
FS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['S']), disulphide=0.0, pi_stack=0.0)
FT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['T']), disulphide=0.0, pi_stack=0.0)
FV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['V']), disulphide=0.0, pi_stack=0.0)
FW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['W']), disulphide=0.0, pi_stack=0.9)
FY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Y']), disulphide=0.0, pi_stack=0.9)

aa = 'G'
GA = AG
GC = CG
GD = DG
GE = EG
GF = FG
GG = sc_comp(fit=1.0, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd[aa]), disulphide=0.0, pi_stack=0.0)
GH = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['H']), disulphide=0.0, pi_stack=0.0)
GI = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['I']), disulphide=0.0, pi_stack=0.0)
GK = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['K']), disulphide=0.0, pi_stack=0.0)
GL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['L']), disulphide=0.0, pi_stack=0.0)
GM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['M']), disulphide=0.0, pi_stack=0.0)
GN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['N']), disulphide=0.0, pi_stack=0.0)
GP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['P']), disulphide=0.0, pi_stack=0.0)
GQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Q']), disulphide=0.0, pi_stack=0.0)
GR = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['R']), disulphide=0.0, pi_stack=0.0)
GS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['S']), disulphide=0.0, pi_stack=0.0)
GT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['T']), disulphide=0.0, pi_stack=0.0)
GV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['V']), disulphide=0.0, pi_stack=0.0)
GW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['W']), disulphide=0.0, pi_stack=0.0)
GY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Y']), disulphide=0.0, pi_stack=0.0)

aa = 'H'
HA = AH
HC = CH
HD = DH
HE = EH
HF = FH
HG = GH
HH = sc_comp(fit=0.8, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd[aa]), disulphide=0.0, pi_stack=0.0)
HI = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['I']), disulphide=0.0, pi_stack=0.0)
HK = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['K']), disulphide=0.0, pi_stack=0.0)
HL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['L']), disulphide=0.0, pi_stack=0.0)
HM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['M']), disulphide=0.0, pi_stack=0.0)
HN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['N']), disulphide=0.0, pi_stack=0.0)
HP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['P']), disulphide=0.0, pi_stack=0.0)
HQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Q']), disulphide=0.0, pi_stack=0.0)
HR = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['R']), disulphide=0.0, pi_stack=0.0)
HS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['S']), disulphide=0.0, pi_stack=0.0)
HT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['T']), disulphide=0.0, pi_stack=0.0)
HV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['V']), disulphide=0.0, pi_stack=0.0)
HW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['W']), disulphide=0.0, pi_stack=0.0)
HY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Y']), disulphide=0.0, pi_stack=0.0)

aa = 'I'
IA = AI
IC = CI
ID = DI
IE = EI
IF = FI
IG = GI
IH = HI
II = sc_comp(fit=0.8, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd[aa]), disulphide=0.0, pi_stack=0.0)
IK = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['K']), disulphide=0.0, pi_stack=0.0)
IL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['L']), disulphide=0.0, pi_stack=0.0)
IM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['M']), disulphide=0.0, pi_stack=0.0)
IN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['N']), disulphide=0.0, pi_stack=0.0)
IP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['P']), disulphide=0.0, pi_stack=0.0)
IQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Q']), disulphide=0.0, pi_stack=0.0)
IR = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['R']), disulphide=0.0, pi_stack=0.0)
IS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['S']), disulphide=0.0, pi_stack=0.0)
IT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['T']), disulphide=0.0, pi_stack=0.0)
IV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['V']), disulphide=0.0, pi_stack=0.0)
IW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['W']), disulphide=0.0, pi_stack=0.0)
IY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd[aa]+hyd['Y']), disulphide=0.0, pi_stack=0.0)

aa = 'K'
KA = AK
KC = CK
KD = DK
KE = EK
KF = FK
KG = GK
KH = HK
KI = IK
KK = sc_comp(fit=0.8, Hbond=0.1, electrostat=-1.0, hydroph=(hyd['K']+hyd['K']), disulphide=0.0, pi_stack=0.0)
KL = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['L']), disulphide=0.0, pi_stack=0.0)
KM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['M']), disulphide=0.0, pi_stack=0.0)
KN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['N']), disulphide=0.0, pi_stack=0.0)
KP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['P']), disulphide=0.0, pi_stack=0.0)
KQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['Q']), disulphide=0.0, pi_stack=0.0)
KR = sc_comp(fit=0.3, Hbond=0.1, electrostat=-1.0, hydroph=(hyd['K']+hyd['R']), disulphide=0.0, pi_stack=0.0)
KS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['S']), disulphide=0.0, pi_stack=0.0)
KT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['T']), disulphide=0.0, pi_stack=0.0)
KV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['V']), disulphide=0.0, pi_stack=0.0)
KW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['W']), disulphide=0.0, pi_stack=0.5)
KY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['Y']), disulphide=0.0, pi_stack=0.5)

LA = AL
LC = CL
LD = DL
LE = EL
LF = FL
LG = GL
LH = HL
LI = IL
LK = KL
LL = sc_comp(fit=0.8, Hbond=0.1, electrostat=0.0, hydroph=(hyd['L']+hyd['L']), disulphide=0.0, pi_stack=0.0)
LM = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['L']+hyd['M']), disulphide=0.0, pi_stack=0.0)
LN = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['L']+hyd['N']), disulphide=0.0, pi_stack=0.0)
LP = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['L']+hyd['P']), disulphide=0.0, pi_stack=0.0)
LQ = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['L']+hyd['Q']), disulphide=0.0, pi_stack=0.0)
LR = sc_comp(fit=0.3, Hbond=0.1, electrostat=-1.0, hydroph=(hyd['K']+hyd['R']), disulphide=0.0, pi_stack=0.0)
LS = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['S']), disulphide=0.0, pi_stack=0.0)
LT = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['T']), disulphide=0.0, pi_stack=0.0)
LV = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['V']), disulphide=0.0, pi_stack=0.0)
LW = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['W']), disulphide=0.0, pi_stack=0.5)
LY = sc_comp(fit=0.3, Hbond=0.1, electrostat=0.0, hydroph=(hyd['K']+hyd['Y']), disulphide=0.0, pi_stack=0.5)