%{
#include "parser.h"
#ifdef DUMP_TOKENS
    #define RETURN(t) do {                                      \
        fprintf ( stderr, "TOKEN ( %d,\t'%s' )\n", t, yytext );    \
        return t;                                               \
    } while ( 0 )
#else
    #define RETURN(t) return t
#endif
%}

%option array
%option noyywrap
%option yylineno

delim 		[ \t\n]
letter		[a-zA-Z_]
character	[^\n]
ws 			{delim}+
comment		\/\/.*\n
digit		[0-9]
number		{digit}*
string		\"([^\n\"]|\\\")*\"
id 			{letter}({letter}|{digit})*
ge 			>=
le 			<=			

%%

{ws}		{/**/}
{comment}	{/**/}
FUNC 		{ RETURN(FUNC); }
PRINT		{ RETURN(PRINT); }		
CONTINUE 	{ RETURN(CONTINUE); }
IF 		 	{ RETURN(IF); }
ELSE 	 	{ RETURN(ELSE); }
FI  	 	{ RETURN(FI); }
WHILE 	 	{ RETURN(WHILE); }
DO  	 	{ RETURN(DO); }
DONE 	 	{ RETURN(DONE); }
FOR  	 	{ RETURN(FOR); }
TO  	 	{ RETURN(TO); }
VAR  	 	{ RETURN(VAR); }
THEN		{ RETURN(THEN); }
{number}	{ RETURN(NUMBER); }
{string}	{ RETURN(STRING); }
RETURN  	{ RETURN(RETURN); }
{id}		{ RETURN(IDENTIFIER); }
:=			{ RETURN(ASSIGN); }
==			{ RETURN(EQUAL); }
{le}		{ RETURN(LEQUAL); }
{ge}		{ RETURN(GEQUAL); }
!=			{ RETURN(NEQUAL); }
.			{ RETURN( yytext[0] ); }

%%

