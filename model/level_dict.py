'''
#version 2-3-17#'ALSR FAIL','M1-ABNORMAL','M1-PARTICLE','M2-ABNORMAL','M2-PARTICLE'
hierarchy = {
    'TFT':['NP','OP','UP'],
    'CF':['NP','OP','UP']
}

hierarchy_two = {
    'NP':['ALCV FAIL','AS-RESIDUE-E','AS-RESIDUE-T','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','ITO-ABNORMAL','LIGHT METAL','M2-RESIDUE-P','POLYMER'],
    'OP':['CELL REPAIR FAIL','FIBER','GLASS CULLET','LIGHT METAL','POLYMER','SPI-POLYMER','V-POLYMER'],
    'UP':['FIBER','GLASS CULLET','LIGHT METAL','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER']
}
'''

'''
#version 2-3-14 #'ALSR FAIL','M1-ABNORMAL','M1-PARTICLE','M2-ABNORMAL','M2-PARTICLE','AS-RESIDUE-T','SPI-POLYMER','V-POLYMER'
hierarchy = {
    'TFT':['NP','OP','UP'],
    'CF':['NP','OP','UP']
}

hierarchy_two = {
    'NP':['ALCV FAIL','AS-RESIDUE-E','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','ITO-ABNORMAL','LIGHT METAL','M2-RESIDUE-P','POLYMER'],
    'OP':['CELL REPAIR FAIL','FIBER','GLASS CULLET','LIGHT METAL','POLYMER'],
    'UP':['FIBER','GLASS CULLET','LIGHT METAL','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER']
}
'''



#version 2-3-16#'ALSR FAIL','M1-ABNORMAL','M1-PARTICLE','M2-ABNORMAL','M2-PARTICLE','AS-RESIDUE-T'
hierarchy = {
    'TFT':['NP','OP','UP'],
    'CF':['NP','OP','UP']
}

hierarchy_two = {
    'NP':['ALCV FAIL','AS-RESIDUE-E','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','ITO-ABNORMAL','LIGHT METAL','M2-RESIDUE-P','POLYMER'],
    'OP':['CELL REPAIR FAIL','FIBER','GLASS CULLET','LIGHT METAL','POLYMER','SPI-POLYMER','V-POLYMER'],
    'UP':['FIBER','GLASS CULLET','LIGHT METAL','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER']
}




'''
#version 2-3-22#SPI-POLYMER','V-POLYMER'
hierarchy = {
    'TFT':['NP','OP','UP'],
    'CF':['NP','OP','UP']
}

hierarchy_two = {
    'NP':['ALCV FAIL','ALSR FAIL','AS-RESIDUE-E','AS-RESIDUE-T','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','ITO-ABNORMAL','LIGHT METAL','M1-ABNORMAL','M1-PARTICLE','M2-ABNORMAL','M2-PARTICLE','M2-RESIDUE-P','POLYMER'],
    'OP':['CELL REPAIR FAIL','FIBER','GLASS CULLET','LIGHT METAL','POLYMER','SPI-POLYMER','V-POLYMER'],
    'UP':['FIBER','GLASS CULLET','LIGHT METAL','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER']
}
'''



'''version 2-3-26'''
'''
hierarchy = {
    'TFT':['NP','OP','UP'],
    'CF':['NP','OP','UP']
}

hierarchy_two = {
    'NP':['ALCV FAIL','ALSR FAIL','AS-RESIDUE-E','AS-RESIDUE-T','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','ITO-ABNORMAL','ITO-RESIDUE-P','ITO-RESIDUE-T','LIGHT METAL','M1-ABNORMAL','M1-PARTICLE','M2-ABNORMAL','M2-PARTICLE','M2-RESIDUE-P','POLYMER','PV-HOLE-T'],
    'OP':['CELL REPAIR FAIL','FIBER','GLASS CULLET','LIGHT METAL','PE','POLYMER','SPI-POLYMER','V-POLYMER'],
    'UP':['FIBER','GLASS CULLET','LIGHT METAL','PI SPOT-NO PAR','PI SPOT-WITH PAR','POLYMER']
}
'''


'''version 2-3-12
'''
'''
hierarchy = {
    'TFT':['NP','OP','UP'],
    'CF':['NP','OP','UP']
}

hierarchy_two = {
    'NP':['AS-RESIDUE-E','CF DEFECT','CF PS DEFORMATION','CF REPAIR FAIL','ITO-RESIDUE-T','LIGHT METAL','M1-ABNORMAL','POLYMER','PV-HOLE-T'],
    'OP':['POLYMER','FIBER','GLASS CULLET','LIGHT METAL'],
    'UP':['POLYMER','FIBER','GLASS CULLET','LIGHT METAL','PI SPOT-WITH PAR']
}
'''



'''version 1;2-4-14

hierarchy = {
    'TFT':['NP','OP','UP','INT'],
    'CF':['NP','OP','UP']
}

hierarchy_two = {
    'NP':['CF REPAIR FAIL','PV-HOLE-T','CF DEFECT','CF PS DEFORMATION','AS-RESIDUE-E','ITO-RESIDUE-T','M1-ABNORMAL'],
    'OP':['POLYMER','FIBER','GLASS CULLET'],
    'UP':['PI SPOT-WITH PAR','LIGHT METAL','FIBER','POLYMER'],
    'INT' : ['GLASS BROKEN','ESD']
}

'''







'''Dictionary for CIFAR-100 hierarchy.
'''

'''
hierarchy = {
    'aquatic_mammals':['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish':	['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers':['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers' : ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables':['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices' :['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture':	['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects':	['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores':['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things':['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes':['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores' :	['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates':	['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people':	['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees' :	['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1':['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}
'''