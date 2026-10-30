"""Publication-independent diagnostic plots of inner scores only."""
from pathlib import Path
import json
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];O=R/'outputs/er_inner_convergence_20261030';S=O/'extracted/results/study'
def main():
 im=Image.new('RGB',(1000,1650),'white');d=ImageDraw.Draw(im)
 d.text((30,15),'Inner-validation pseudo-label Dice vs updates | 5 specimen-isolated folds',fill='black')
 d.text((30,38),'Blue: original pool | Orange: expanded pool | Circles: locked selected checkpoints',fill='black')
 for fid in range(5):
  top=90+fid*300;left=80;right=950;bottom=top+240
  d.text((30,top-22),f'Fold {fid} (inner validation only; not outer test)',fill='black')
  for tick in [0,.25,.5,.75,1]:
   yy=bottom-int(tick*240);d.line((left,yy,right,yy),fill='#dddddd');d.text((40,yy-5),f'{tick:.2f}',fill='black')
  for tick in [0,1000,2000,3000,4000]:
   xx=left+int(tick/4000*(right-left));d.text((xx-12,bottom+8),str(tick),fill='black')
  d.text((440,bottom+25),'Training updates (steps)',fill='black');d.text((3,top+100),'Dice',fill='black')
  for arm,color in [('control','#2363aa'),('expanded','#c76b12')]:
   h=json.loads((S/f'fold{fid}'/arm/'history.json').read_text());choice=json.loads((S/f'fold{fid}'/arm/'decision.json').read_text())
   points=[(left+int(v['step']/4000*(right-left)),bottom-int(v['inner_dice']*240)) for v in h];d.line(points,fill=color,width=3)
   x=left+int(choice['selected_step']/4000*(right-left));y=bottom-int(choice['inner_dice']*240);d.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3)
 d.text((30,1610),'Source: frozen ER inner-convergence study, 2026-10-30. Same automatic binary labels; not manual accuracy.',fill='black')
 im.save(O/'inner_learning_curves.png')
if __name__=='__main__':main()
