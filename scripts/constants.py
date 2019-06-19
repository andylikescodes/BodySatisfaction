# Useful categories
cat_demo = ['Sex', 'SexualOrientationX4', 'EthnicityX5', 'RelationshipStatus','CollegeStudent']
num_demo = ['BMI', 'Age', 'LevelEducation']
cont_avg = ['AETOTAL', 'SATAQThinInternalization', 'SATAQMuscleInternalization', 'SATAQFamilyPressure', 
            'SATAQPeerPressure', 'SATAQMediaPressure', 'FaceSatisfactionTotal', 'OverweightPreoccupationTotal', 
            'BIQLITotal', 'SURVEILLANCETotal']
survey_data_aggregate = ['SATAQThinInternalization', 'SATAQMuscleInternalization', 'SATAQFamilyPressure', 
            'SATAQPeerPressure', 'SATAQMediaPressure', 'FaceSatisfactionTotal', 'SURVEILLANCETotal']

survey_data_raw = ['SATAQThinInternalization', 'SATAQMuscleInternalization',
       'SATAQFamilyPressure', 'SATAQPeerPressure', 'SATAQMediaPressure',
       'ThinSATAQ1BodyThin', 'ThinSATAQ2ThinkThin', 'ThinSATAQ3BodyLean',
       'ThinSATAQ4LittleFat', 'MuscleSATAQ1AthleticImportant',
       'MuscleSATAQ2Muscular', 'MuscleSATAQ3AthleticThings',
       'MuscleSATAQ4ThinkAthletic', 'MuscleSATAQ5MuscularThings',
       'FamilySATAQ1PressureThin', 'FamilySATAQ2PressureAppearance',
       'FamilySATAQ3DecreaseBodyFat', 'FamilySATAQ4BetterShape',
       'PeersSATAQ1Thinner', 'PeersSATAQ2ImproveAppearance',
       'PeersSATAQ3BetterShape', 'PeersSATAQ4DecreaseBodyFat',
       'MediaSATAQ1BetterShape', 'MediaSATAQ2Thinner',
       'MediaSATAQ3ImproveAppearance', 'MediaSATAQ4decreaseBodyFat',
       'FaceSatisfaction1HappyFace', 'FaceSatisfaction2HappyNose',
       'FaceSatisfaction3HappyEyes', 'FaceSatisfaction4HappyShape',
       'Surveillance1ThinkAboutLooksRECODED',
       'Surveillance2ComfortableClothesRECODED',
       'Surveillance3BodyFeelsOverLooksRECODED',
       'Surveillance4CompareLooksRECODED', 'Surveillance5LooksDuringDay',
       'Surveillance6WorryClothes', 'Surveillance7LooksToOtherPeopleRECODED',
       'Surveillance8BodyDoesBodyLooksRECODED']

y_variables = ['AETOTAL', 'OverweightPreoccupationTotal', 
               'OverweightPreoccupation3Diet', 'OverweightPreoccupation4TriedFasting',
               'BIQLITotal']

# interation setting
data_sets = {'aggregate': survey_data_aggregate, 'raw': survey_data_raw}