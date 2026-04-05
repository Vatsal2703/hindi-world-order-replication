#include <stdio.h>
#include "commandline.h"
#include "unicode.h"
#include "tagfile.h"
#include "rule.h"


/* unirule input_filename output_filename rulelist_filename passes */
main(int argc, char *argv[])
{
	rule *rulelist;
	int i, iterations;

	char *infile;
	char *outfile = "___temp_unirulefileOUT.txt";

	/* variables for counting the rules */
	rule *current;
	int rulecount;
	char b36rc[20];
	unichar ub36rc[20];




	/* check arguments */
	if (argc != 5)
	{
		cl_error_arguments(5, argc);
		return 1;
	}


	/* load the rulelist */
	rulelist = load_rulelist(argv[3]);

	if (rulelist == NULL)
	{
		puts("No rulelist loaded!");
		return 1;
	}



	/* quick report on how many rules have been loaded */
	rulecount = 1;
	for ( current = rulelist ; current->nextrule ; current = current->nextrule )
		rulecount++;
	ub36rc[0] = 0;
	assign_ruleresp(ub36rc, rulecount);
	ub36rc[3] = 0;
	for ( i = 0 ; i < 20 ; i++ )
		b36rc[i] = (char)ub36rc[i];
	printf("Unirule has successfully loaded %d rules (up to %s).\n", rulecount, b36rc);



	/* acquire a number of iterations */
	sscanf(argv[4], "%d", &iterations);



	/* run the function over appropriate number of iterations */
	for ( i = 0 ; i < iterations ; i++)
	{
		if (i == 0)
			infile = argv[1];
		else
		{
			infile = "___temp_unirulefileIN.txt";
			rename(outfile, infile);
		}
		if ( apply_rules_file(rulelist, infile, outfile) )
		{
			puts("An error was reported in processing this file.");
			fcloseall();
			free_rulelist(rulelist);
			return 1;
		}
		if (infile != argv[1])
			remove(infile);
	}
	rename(outfile, argv[2]);

	return 0;
}