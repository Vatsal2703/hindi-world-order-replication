/* rustrident */
int rustrident(unichar *string1, unichar *string2);


/* the assess functions */

int assess_ifthiswordis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifthistagis(token *thisword,
					   token *prevword[RULESPAN],
					   token *nextword[RULESPAN],
					   short int r,
					   unichar *matchstring);

int assess_ifthistaginc(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifprevwordis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifprevtagis(token *thisword,
					   token *prevword[RULESPAN],
					   token *nextword[RULESPAN],
					   short int r,
					   unichar *matchstring);

int assess_ifprevtaginc(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifnextwordis(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifnexttagis(token *thisword,
					   token *prevword[RULESPAN],
					   token *nextword[RULESPAN],
					   short int r,
					   unichar *matchstring);

int assess_ifnexttaginc(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

/* NEGATIVE  assess functions */

int assess_ifthiswordisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);


int assess_ifthistagisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifthistagincnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifprevwordisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifprevtagisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifprevtagincnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifnextwordisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifnexttagisnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);

int assess_ifnexttagincnot(token *thisword,
						token *prevword[RULESPAN],
						token *nextword[RULESPAN],
						short int r,
						unichar *matchstring);