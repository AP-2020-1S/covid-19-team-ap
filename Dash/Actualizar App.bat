call activate 
call python.exe casos.py 
call python.exe "Actualizar modelos.py"
call heroku container:login 
call heroku container:push web -a teamap-unal 
call heroku container:release web -a teamap-unal 
