# pip install psychopy
# 运动图像生成
from psychopy import visual,core,event
import random
import numpy as np

win = visual.Window(size=(1000,600),color=(-1,-1,-1),fullscr=False)

text_1 = visual.TextStim(win, text = 'ready?',
                               height = 0.5,
                               pos = (0.0,0.2),
                               color = 'pink',
                               bold = True,
                               italic = True)

num = np.ones(50,dtype=np.int)
num[:25] = np.zeros_like(num[:25])
np.random.shuffle(num)


length = 50
text_1.draw()
win.flip()
core.wait(0)
k_1 = event.waitKeys()
timer = core.Clock()
for i in range(length):
    text_1.text = '+'
    text_1.draw()
    win.flip()
    core.wait(2)

    tem = num[i]
    if tem == 1:
        text_1.text = '←'
        text_1.draw()
        win.flip()
        core.wait(6)
    else:
        text_1.text = '→'
        text_1.draw()
        win.flip()
        core.wait(6)

    text_1.text = ' '
    text_1.draw()
    win.flip()
    core.wait(2)
    


win.close()
core.quit()

