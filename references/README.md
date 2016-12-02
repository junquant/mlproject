Useful Links

### Git workflow
* Git Workflow [here] (https://www.atlassian.com/git/tutorials/comparing-workflows/centralized-workflow)
* We are probably following the centralized workflow. 

### Markdown (.md) to Microsoft Docx
* [pandoc](http://pandoc.org/installing.html) - [Download Link](http://pandoc.org/installing.html)
* Help command : pandoc --help
* Command to convert md to docx : 
    * pandoc proposal.md -f markdown -t docx -o output.docx (Works for OSX and Windows)
	* where -f is from
	* -t is to
	* -o is output file name
* Command to convert md to html slides (reveal.js) : 
    * pandoc -s --mathjax -t revealjs sample_deck.md -o sample_deck.html
	* where -s is standalone
	* -t is to 
	* sample_deck.md is source file name
	* -o is output file name
* For reveal.js, the folder reveal.js needs to be in the same folder as the html slide deck

### Handling Missing Data

* [Dealing with Missing Data by Marina Soley-Bori] (http://www.bu.edu/sph/files/2014/05/Marina-tech-report.pdf)
* [Missing Data and How to Deal: An Overview of Missing Data] (https://liberalarts.utexas.edu/prc/_files/cs/Missing-Data.pdf)

### Algorithms

* Naive Bayes
  * [Naive Bayes Tutorial for ML] (http://machinelearningmastery.com/naive-bayes-tutorial-for-machine-learning/)
  * [Naive Bayes Classifier from Scratch in Python] (http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)
* Support Vector Machine
  * [Support Vector Machine for ML] (http://machinelearningmastery.com/support-vector-machines-for-machine-learning/)
* Classification and Regression Tree (CART)
 * [Classification And Regression Trees for ML] (http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)

### Numpy Stuff
* Structured arrays [here] (http://docs.scipy.org/doc/numpy/user/basics.rec.html)
