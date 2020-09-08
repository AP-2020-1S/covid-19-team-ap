call C:\Users\Julian\anaconda3\python.exe C:\Users\Julian\anaconda3\Scripts\activate casos.py
call C:\Users\Julian\anaconda3\python.exe C:\Users\Julian\anaconda3\Scripts\activate "Actualizar modelos.py"
call heroku container:login 
call heroku container:push web -a teamap-unal 
call heroku container:release web -a teamap-unal 
