## Load some data and define MABs

module NewYorker

export load_probas, probas_contest499, captions_contest499

import DataFrames

## Load multinomial data

function loadData(dir, contest)
    pathprobas = "$dir/probabilities_$contest.csv"
    pathcaptions = "$dir/captions_$contest.txt" 

    probas = DataFrames.readtable(pathprobas)
    captions = DataFrames.readtable(pathcaptions)

    return convert(Array, probas), convert(Array, captions)
end
   
contests_dir = "../../resources"
contest = 499

probas_contest499, captions_contest499 = loadData(contests_dir, contest)

end
