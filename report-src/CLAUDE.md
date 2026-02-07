@convNPClimate/docs/DAS8_DONNER.pdf @convNPClimate/docs/Vaughan-2021.pdf 

This capstone project was defined by Prof. Donner in DAS8_DONNER.pdf. We found the Vaughan 2021 paper and pivoted to trying to build upon it to verify its usefulness, its weaknesses and taking steps towards making it productionized and useful in the real world.

convNPClimate/convCNP contains the Vaughan core code, with minimal changes we added for wiring configuration through and for getting telemetry out.

convNPClimate/*.py contains all the fundamental code we added
convNPClimate/training-notebook.ipynb contains the primary wiring for training that we created and used.
convNPClimate/predictions.ipynb contains the primary tooling we created for prediction and metrics generation.

We are now writing the report in tex in convNPClimate/report-src.