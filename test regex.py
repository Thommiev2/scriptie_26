import os
from pathlib import Path
import csv
from datetime import datetime
import jiwer

gt = "hi hi hi hsodf osd ofisdo fmsd f"
h = "hiss hi hi s a  hodf osd ofisdo fmsd f"
a = jiwer.process_words(gt, h)
print(a)
# def export_transcript(category, model, name, text):
#
# 	a = Path(f'ASR output') / model / category
# 	a.mkdir(parents=True, exist_ok=True)
#
# 	with open(a / name, 'w', encoding='utf-8') as f:
# 		f.write(text)
# 		f.close()

def format_output_as_csv(transcript):
	with open(f"{datetime.now()}.csv", mode="w", newline="", encoding="utf-8") as file:
		a = 'f'

# b = " Goedemiddag. Wat heeft u hier gebracht vandaag? Ik heb de laatste tijd wat last van mijn hart. Uw hart. En wat bedoelt u daarmee? Nou, ik merk dat ik steeds vaker hartkloppingen heb. Oké. En hoe lang heeft u daar al last van? Ik denk een paar weken, maar sommige dagen heb ik er ook helemaal geen last van. Dus eigenlijk heeft u bijna iedere dag er last van en wat voelt u dan precies? Mijn hart klopt dus sneller en ik voel het echt in mijn borstkas kloppen. Oké, dat is vast een onprettig gevoel. U zegt dat u het al een paar weken heeft, daarvoor had u het niet. Wat maakt nou dat u vandaag bent gekomen? Ja, het is gewoon een heel naar gevoel en ik denk dat ik een hartaanval krijg ofzo, dus ik wil het in en ander laten checken. Ja, dat begrijp ik helemaal. En op zo'n moment van de aanval, wat doet u dan? Ja, niks speciaals eigenlijk. Voelt u verder nog andere dingen op die momenten in de borstkast of in de rest van je lichaam?"
#

