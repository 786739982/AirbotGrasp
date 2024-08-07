from AirbotGrasper import AirbotGrasper

if __name__ == '__main__':
    
    airbot_grasper = AirbotGrasper(visuale_model='SAM', debug=False)
    while 1:
        airbot_grasper.run()