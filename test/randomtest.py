from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


inp = [128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,220,2366,18,198,15724,2696,25,220,845,3799,220,2366,19,271,7927,3465,374,311,11886,264,37072,3575,3094,555,3094,13,1789,904,
22702,11,1005,279,7891,364,2501,2511,6,311,79164,279,22702,13,1789,3187,11,422,499,1390,311,11294,220,17,10,17,11,499,1288,3350,1134,17,10,17,28,19,2511,19,13,17830,11,842,701,6425,449,1595,791,4320,374
,25,510,9399,60,19154,8144,1475,3094,315,279,6425,304,264,502,1584,13,128009,128006,882,128007,271,26597,706,1403,20820,13,5414,24417,10868,374,220,16,1060,9191,1109,11157,8096,596,4325,994,8096,574,264
,1060,14992,13,5414,14992,10868,374,220,20,1667,2362,11,902,374,264,4948,315,279,4325,315,279,9191,10868,13,3639,374,872,11093,4325,30,128009,128006,78191,128007,271,791,24417,10868,374,220,868,1606,220
,18,865,220,20,284,1134,18,9,20,28,868,2511,868,198,26597,374,220,23,1606,422,220,16,489,320,26597,596,4325,482,220,16,8,865,220,17,284,220,868,1243,8096,596,4325,284,220,23,198,35897,11093,4325,374,220
,1591,1606,220,20,489,220,868,489,220,23,284,1134,20,10,868,10,23,28,1591,2511,1591,198,791,4320,374,25,220,1591,
128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009,128009]


labels =  [-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100
,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,
           -100,791,24417,10868,374,220,868,1606,220,18,865,220,20,284,1134,18,9,20,28,868,2511,868,198,26597,374,220,23,1606,422,220,16,489,320,26597,596,4325,482,220,16,8,865,220,
17,284,220,868,1243,8096,596,4325,284,220,23,198,35897,11093,4325,374,220,1591,1606,220,20,489,220,868,489,220,23,
           284,1134,20,10,868,10,23,28,1591,2511,1591,198,791,4320,374,25,220,1591,128009,-100,-100
,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100]

print(len(inp), len(labels))
padlabels = []

for l in labels:
    if l == -100:
        padlabels.append(tok.eos_token_id)
    else:
        padlabels.append(l)

print("******")
print(tok.decode(inp))
print("******")
print(tok.decode(padlabels))


inp = [128000,128006,9125,128007,271,38766,1303,33025,2696,25,6790,220,2366,18,198,15724,2696,25,220,845,3799,220,2366,19,271,7927,3465,374,311,11886,264,37072,3575,3094,555,3094,13,1789,904,
22702,11,1005,279,7891,364,2501,2511,6,311,79164,279,22702,13,1789,3187,11,422,499,1390,311,11294,220,17,10,17,11,499,1288,3350,1134,17,10,17,28,19,2511,19,13,17830,11,842,701,6425,449,1595,791,4320,374
,25,510,9399,60,19154,8144,1475,3094,315,279,6425,304,264,502,1584,13,128009,128006,882,128007,271,26597,706,1403,20820,13,5414,24417,10868,374,220,16,1060,9191,1109,11157,8096,596,4325,994,8096,574,264
,1060,14992,13,5414,14992,10868,374,220,20,1667,2362,11,902,374,264,4948,315,279,4325,315,279,9191,10868,13,3639,374,872,11093,4325,30,128009,128006,78191,128007,271,791,24417,10868,374,220,868,1606,220
,18,865,220,20,284,1134,18,9,20,28,868,2511,868,198,26597,374,220,23,1606,422,220,16,489,320,26597,596,4325,482,220,16,8,865,220,17,284,220,868,1243,8096,596,4325,284,220,23,198,35897,11093,4325,374,220
,1591,1606,220,20,489,220,868,489,220,23,284,1134,20,10,868,10,23,28,1591,2511,1591,198,791,4320,374,25,220,1591,128009]


labels =  [-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100
,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,
-100,-100,791,24417,10868,374,220,868,1606,220,18,865,220,20,284,1134,18,9,20,28,868,2511,868,198,26597,374,220,23,1606,422,220,16,489,320,26597,596,4325,482,220,16,8,865,220,
17,284,220,868,1243,8096,596,4325,284,220,23,198,35897,11093,4325,374,220,1591,1606,220,20,489,220,868,489,220,23,284,1134,20,10,868,10,23,28,1591,2511,1591,198,791,4320,374,25,220,1591,128009]

print(len(inp), len(labels))
padlabels = []

for l in labels:
    if l == -100:
        padlabels.append(tok.eos_token_id)
    else:
        padlabels.append(l)

print("******")
print(tok.decode(inp))
print("******")
print(tok.decode(padlabels))