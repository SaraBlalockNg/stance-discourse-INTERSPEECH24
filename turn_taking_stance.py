#!/usr/bin/env python3

import argparse
import copy
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from io import StringIO
from matplotlib.ticker import MaxNLocator
from pandas.errors import SettingWithCopyWarning
from scipy import stats
from scipy.stats import norm
from utils import *


# these are the sessions in ATAROS that we assume will prcess correctly
pairs = ['NWM054-NWF092-6B_','NWM055-NWF093-6B_','NWF089-NWM053-3I_',
		'NWM029-NWM065-6B_','NWM059-NWF101-6B_', 'NWM061-NWM060-6B_',
		'NWM062-NWM063-6B_', 'NWM064-NWM008-6B_', 'NWM067-NWM047-6B_',
		'NWM068-NWF113-6B_', 'NWM069-NWF114-6B_', 'NWM073-NWF122-6B_',
		'NWM074-NWF123-6B_', 'NWM075-NWF126-6B_', 'NWF090-NWF091-3I_',
		'NWF094-NWF095-3I_', 'NWF096-NWM056-3I_', 'NWF097-NWM057-3I_',
		'NWF098-NWM058-3I_', 'NWF099-NWF100-3I_', 'NWF102-NWF103-3I_',
		'NWF105-NWF104-3I_', 'NWF106-NWF107-3I_', 'NWF108-NWF109-3I_',
		'NWF110-NWM066-3I_', 'NWF111-NWF112-3I_', 'NWF115-NWM070-3I_',
		'NWF116-NWM071-3I_', 'NWF117-NWF118-3I_', 'NWF119-NWM072-3I_',
		'NWF120-NWF121-3I_', 'NWF124-NWF125-3I_', 'NWF127-NWF128-3I_',
		'NWF129-NWF130-3I_', 'NWM029-NWM065-3I_', 'NWM054-NWF092-3I_',
		'NWM055-NWF093-3I_', 'NWM059-NWF101-3I_', 'NWM061-NWM060-3I_',
		'NWM062-NWM063-3I_', 'NWM064-NWM008-3I_', 'NWM067-NWM047-3I_',
		'NWM068-NWF113-3I_', 'NWM069-NWF114-3I_', 'NWM073-NWF122-3I_',
		'NWM074-NWF123-3I_', 'NWM075-NWF126-3I_', 'NWF124-NWF125-6B_',
		'NWF089-NWM053-6B_', 'NWF090-NWF091-6B_', 'NWF094-NWF095-6B_',
		'NWF096-NWM056-6B_', 'NWF097-NWM057-6B_', 'NWF098-NWM058-6B_',
		'NWF099-NWF100-6B_', 'NWF102-NWF103-6B_', 'NWF105-NWF104-6B_',
		'NWF106-NWF107-6B_', 'NWF108-NWF109-6B_', 'NWF110-NWM066-6B_',
		'NWF111-NWF112-6B_', 'NWF115-NWM070-6B_', 'NWF116-NWM071-6B_',
		'NWF117-NWF118-6B_', 'NWF119-NWM072-6B_', 'NWF120-NWF121-6B_',
		'NWF127-NWF128-6B_','NWF129-NWF130-6B_'] 

bk_words =	['okay', 'oh', 'i see', 'uh-huh', 'yeah', 'all right', 'uh', 
			'right', 'um', 'huh', 'yes', 'oh okay', 'no', 'ooh', 'well']

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class NewTurnData():
	def __init__(self,speaker1,speaker2,stance_df,bk_words=bk_words):
		self.IPUS = {speaker1:[],speaker2:[]}
		self.turn_taking = None
		self.turns = None
		self.pauses = None
		self.stance_df = stance_df
		self.bk_words = bk_words
	
	def add_speaker(self,turn):
		turns_list = []
		ipus_list = []
		for IPU in turn.IPUS:
			self.IPUS[turn.speaker].append([
					IPU.start_time,IPU.end_time,turn.speaker,IPU.text])
	
	def bk_match(self,text):
		splittext = re.split(r'\s',text.lower())
		for i, word in enumerate(splittext):
			if word not in bk_words:
				if ' '.join(splittext[i:i+2]) not in bk_words:
					return False
		return True
	
	def max_dict(self,dic):
		l = [(k,v) for k,v in dic.items()]
		best = max(l,key=lambda x:len(x[1]))
		return best[0]
	
	def max_list(self,l):
		m = 0
		for item in l:
			try:
				i = int(item)
			except Exception:
				continue
			m = max(m,i)
		return m
	
	def find_polarity_and_strength(self,s):
		polarity = re.findall(r'[\-+x]',s)
		strength = re.findall(r'[0123x]',s)
		polarity = polarity[0] if len(polarity)>0 else 'None'
		strength = strength[0] if len(strength)>0 else 0
		return polarity, strength
	
	def flatten_stance(self,stance_list):
		polarity = {'-':[],'+':[],'None':[],'x':[]}
		for candidate in stance_list:
			if len(candidate) in [1,2]:
				this_polarity,this_strength = self.find_polarity_and_strength(
						candidate)
				polarity[this_polarity].append(this_strength)
			else:
				polarity['None'].append(candidate)
		best_key = self.max_dict(polarity)
		strength = self.max_list(polarity[best_key])
		return f'{strength}{best_key if best_key!="None" else ""}'
	
	def check_for_stance(self,row):
		start_time = row.start_time
		end_time = row.end_time
		speaker = row.speaker
		# x is for stance that starts within the time window
		x = self.stance_df.loc[(self.stance_df.start_time >= start_time)&(
				self.stance_df.start_time < end_time)&(
				self.stance_df.speaker.eq(speaker))]
		# y is for stance that end within the time window
		y = self.stance_df.loc[(self.stance_df.end_time <= end_time )&(
				self.stance_df.end_time > start_time)&(
				self.stance_df.speaker.eq(speaker))]
		# z for stance that contains the time window 
		# (this happens because stance is coarse-aligned)
		z = self.stance_df.loc[(self.stance_df.start_time <= start_time)&(
				self.stance_df.end_time >= end_time)&(
				self.stance_df.speaker.eq(speaker))]
		candidates = pd.merge(x,pd.merge(y,z,how='outer'),how='outer')
		if candidates.shape[0]>0:
			if args.flatten_stance:
				return self.flatten_stance(candidates.stance.values)
			else:
				return '|'.join(sorted(set(candidates.stance.values)))
		else:
			return ""
	
	def interleave_turns(self):
		self.IPUS =  {speaker:pd.DataFrame(IPUS, columns=['start_time',
				'end_time', 'speaker', 'text']
				) for speaker,IPUS in self.IPUS.items()}
		self.IPUS = pd.concat(self.IPUS.values()).sort_values(
				by=['start_time','end_time']).reset_index(drop=True)
		if hasattr(self.stance_df,'shape'):
			self.IPUS['stance'] = self.IPUS.apply(self.check_for_stance,
					axis=1)
		self.IPUS['duration']= self.IPUS.apply(lambda x:x.end_time-x.start_time,
				axis=1)
		if args.load_stance and args.flatten_stance:
			self.IPUS['stance_list'] = self.IPUS.stance.apply(lambda x:[x])

		# initialize everything
		turns = []
		current_turn = copy.deepcopy(self.IPUS.iloc[0])
		previous = copy.deepcopy(self.IPUS.iloc[0])
		turn_taking = []

		to_exclude = []
		for i, row in self.IPUS.iterrows():
			if i == 0:
				continue
			if row.speaker == previous.speaker:
				# if theres a negative FTO we have a problem
				assert(row.start_time > previous.end_time)
				# if the next row is same as the last, its a pause
				# if it has stance, it should be the stance of the both
				field = [row.start_time - previous.end_time,
						previous.speaker,row.speaker,'OverlapW',
						previous.end_time,row.start_time]
				if args.load_stance:
					field += [previous.stance,row.stance]
				turn_taking.append(field)
				
				# add to the working turn
				current_turn.end_time = row.end_time
				current_turn.text += " | " + row.text
				if args.load_stance:
					if args.flatten_stance:
						if row.stance and len(row.stance)>0:
							current_turn.stance_list += row.stance_list
					else:
						current_turn.stance += ("|" + row.stance) if (
								row.stance) else ""
			else:
				# its either a backchannel, interruption, or gap
				# we will know if it's a backchannel if its end time is smaller
				if row.end_time > previous.end_time:
					# its an overlapB or gap
					if args.filter_bk and (row.start_time <= previous.end_time
							) and self.bk_match(row.text):
						# instead of an overlapB, treat it as a backchannel
						field = [row.end_time-row.start_time,
								previous.speaker,row.speaker,'Backchannel',
								row.start_time,row.end_time]
						if args.load_stance:
							field += [previous.stance,row.stance]		
						turn_taking.append(field)
						to_exclude.append(i)
						continue
					field = [row.start_time - previous.end_time,
							previous.speaker,row.speaker,'FTO',
							min(row.start_time,previous.end_time),
							max(row.start_time,previous.end_time)]
					if args.load_stance:
						field += [previous.stance,row.stance]				
					turn_taking.append(field)
					# reset turns
					turns.append(current_turn)
					current_turn = copy.deepcopy(row)
				else:
					if ((args.filter_bk and self.bk_match(row.text)
						) or not args.filter_bk):
						# it's a backchannel
						field = [row.end_time-row.start_time,
								previous.speaker,row.speaker,'Backchannel',
								row.start_time,row.end_time]
						if args.load_stance:
							field += [previous.stance,row.stance]	
						turn_taking.append(field)
						to_exclude.append(i)
						continue
					else:
						# consider this like an interruption
						# but still don't let it become previous
						field = [row.start_time - previous.end_time,
							previous.speaker,row.speaker,'FTO',
							min(row.start_time,previous.end_time),
							max(row.start_time,previous.end_time)]
						if args.load_stance:
							field += [previous.stance,row.stance]	
						turn_taking.append(field)
					continue
			previous = copy.deepcopy(row)
		turns.append(current_turn) # grab the last one too
		self.IPUS = self.IPUS.loc[~self.IPUS.index.isin(to_exclude)
			].reset_index(drop=True)
		column_names = ['duration','previous_speaker','next_speaker','label',
						 'start_time','end_time']
		if args.load_stance:
			column_names += ['previous_stance','next_stance']
		self.turn_taking = pd.DataFrame(turn_taking,
				columns=column_names)
		column_names =['start_time','end_time',
				'speaker','text']
		if args.load_stance:
			column_names.append('stance')
		self.turns = pd.DataFrame(turns,columns=column_names).sort_values(
				by=['start_time','end_time']).reset_index(drop=True)
		self.turns['duration']= self.turns.apply(
				lambda x:x.end_time-x.start_time,axis=1)
		# map the turn-taking back to the turns to get stance
		if not args.filter_bk:
			assert(self.turn_taking.loc[self.turn_taking.label.eq('FTO')
					].shape[0] + 1 == self.turns.shape[0])
		sub_df = self.turn_taking.loc[self.turn_taking.label.eq('FTO')
				].copy().reset_index()
		unsub_df = self.turn_taking.loc[~self.turn_taking.label.eq('FTO')]
		self.turn_taking = pd.merge(sub_df,unsub_df,how='outer').sort_values(
				by=['start_time','end_time']).reset_index(drop=True)

class Turn:
	def __init__(self,speaker=None,start=None,end=None,text=None):
		self.IPUS = []
		self.speaker = speaker
		self.working_IPU = IPU(start,end,text) if start is not None else None
		self.duration = None

	def is_active(self):
		return self.working_IPU is not None

	def extend_IPU(self,start_time,end_time,text,speaker=None):
		if not self.is_active():
			self.working_IPU = IPU(start_time,end_time,text)
			self.speaker = speaker
		elif start_time - self.working_IPU.end_time >= args.sil_threshold:
			self.working_IPU.finalize() # concatentates IPU text
			self.IPUS.append(self.working_IPU)
			self.working_IPU = IPU(start_time,end_time,text)
		else:
			self.working_IPU.adjust_end_time(text,end_time)

	def finalize(self):
		if self.is_active():
			self.working_IPU.finalize()
			self.IPUS.append(self.working_IPU)

class IPU:
	def __init__(self,start_time,end_time,text):
		self.start_time = start_time
		self.end_time = end_time
		self.text = [text]
		self.duration = end_time - start_time

	def adjust_end_time(self,text,end_time):
		self.end_time = end_time
		self.text.append(text)
		self.duration = self.end_time - self.start_time

	def finalize(self):
		self.text = ' '.join(self.text)

def process_utt_dur_and_fto(dfs,speakers,stance_df=None):
	t = NewTurnData(*speakers,stance_df)
	for df in dfs:
		df = df.sort_values(by=['start_time','end_time'])
		# work on IPUS and turns until they're done and then add them to the next
		current_turn = Turn(df.speaker[0],df.start_time[0],df.end_time[0],df.text[0])
		for i, row in df.iterrows():
			if i ==0:
				continue # ignore the first
			current_turn.extend_IPU(row.start_time,row.end_time,row.text)
		current_turn.finalize()
		t.add_speaker(current_turn)
	t.interleave_turns()
	return t.turn_taking,t.turns,t.IPUS

def load_data():
	textgrid_frame = os.path.join(args.textgrid_dir,'Aligned_{}*.Textgrid')
	turn_dfs = []
	fto_dfs = []
	pause_dfs = []
	backchannel_dfs = []
	IPU_dfs = []

	for pair in pairs:
		try:
			textgrid_path = glob.glob(textgrid_frame.format(pair))[0]
			print(f'Processing {pair}')
		except IndexError:
			print(f'Aligned transcriptions not found for {pair}')
			continue
		task = re.findall(f'6B|3I',textgrid_path)[0]
		interval_dfs, speakers = load_textgrid_intervals(textgrid_path)
		interloc_gender = {speakers[0]:speakers[1][2],
						   speakers[1]:speakers[0][2]}
		if args.load_stance:
			try:				
				prefix = args.single_channel_dir
				stance_paths = (glob.glob(
						f'{prefix}{speakers[0]}-{task}*.TextGrid')+
						glob.glob(f'{prefix}{speakers[1]}-{task}*.TextGrid'))
				stance_df = load_stance_intervals(stance_paths)
				if stance_df is None:
					print(f'stance_failed for {pair}')
					continue
			except Exception:
				print(f'stance not available for {pair}')
				continue
			if args.debug:
				continue
			ftos,turns,IPUS = process_utt_dur_and_fto(interval_dfs,speakers,
					stance_df=stance_df)
		else:
			if args.debug:
				continue
			ftos,turns,IPUS = process_utt_dur_and_fto(interval_dfs,speakers)

		turns['task'] = task
		ftos['task'] = task
		IPUS['task'] = task

		if args.single_gender:
			ftos['gender'] = ftos.next_speaker.str[2]
			turns['gender'] = turns.speaker.str[2]
			IPUS['gender'] = IPUS.speaker.str[2]
		else:
			ftos['gender'] = ftos.next_speaker.str[2]+ftos.previous_speaker.str[2]
			turns['gender']= turns.speaker.apply(lambda x:x[2]+interloc_gender[x])
			IPUS['gender']= IPUS.speaker.apply(lambda x:x[2]+interloc_gender[x])

		pauses = ftos.loc[ftos.label.eq('OverlapW')]
		backchannels = ftos.loc[ftos.label.eq('Backchannel')]
		ftos = ftos.loc[ftos.label.eq('FTO')]

		# Display the DataFrame
		turn_dfs.append(turns)
		fto_dfs.append(ftos)
		pause_dfs.append(pauses)
		backchannel_dfs.append(backchannels)
		IPU_dfs.append(IPUS)
		# import pdb;pdb.set_trace()

	tdf = pd.concat(turn_dfs).reset_index(drop=True)
	idf = pd.concat(IPU_dfs).reset_index(drop=True)
	fdf = pd.concat(fto_dfs).reset_index(drop=True)
	pdf = pd.concat(pause_dfs).reset_index(drop=True)
	bdf = pd.concat(backchannel_dfs).reset_index(drop=True)

	normalized_turns = tdf.loc[(np.abs(stats.zscore(tdf.duration)) < 2)]
	normalized_IPUS = idf.loc[(np.abs(stats.zscore(idf.duration)) < 2)]
	normalized_pauses = pdf.loc[(np.abs(stats.zscore(pdf.duration)) < 2)]
	normalized_ftos = fdf.loc[(np.abs(stats.zscore(fdf.duration)) < 2)]
	normalized_backchannels = bdf.loc[(np.abs(stats.zscore(bdf.duration)) < 2)]

	tdf.to_csv(os.path.join(args.outdir,'turns.csv'),index=False)
	idf.to_csv(os.path.join(args.outdir,'IPUS.csv'),index=False)
	pdf.to_csv(os.path.join(args.outdir,'pauses.csv'),index=False)
	fdf.to_csv(os.path.join(args.outdir,'ftos.csv'),index=False)
	bdf.to_csv(os.path.join(args.outdir,'backchannels.csv'),index=False)
	normalized_turns.to_csv(os.path.join(args.outdir,'normalized_turns.csv'),
			index=False)	
	normalized_IPUS.to_csv(os.path.join(args.outdir,'normalized_IPUS.csv'),
			index=False)
	normalized_pauses.to_csv(os.path.join(args.outdir,'normalized_pauses.csv'),
			index=False)
	normalized_ftos.to_csv(os.path.join(args.outdir,'normalized_ftos.csv'),
			index=False)
	normalized_backchannels.to_csv(os.path.join(args.outdir,
			'normalized_backchannels.csv'),index=False)
	return (idf,tdf,pdf,fdf,bdf,normalized_IPUS,normalized_turns,
			normalized_pauses,normalized_ftos,normalized_backchannels)

def get_strength(x):
	m = {'0':'None','1':'Weak','2':'Moderate','3':'Strong'}
	if type(x)!=str:
		return np.nan
	elif len(x) == 1:
		if args.text_labels:
			return m[x[0]]
		else:
			return x
	elif len(x) == 0:
		return np.nan
	else:
		if args.text_labels:
			return m[x[0]]
		else:
			return x[0]

def get_polarity(x):
	m = {'-':'Negative','+':'Positive'}
	if type(x)!=str:
		return np.nan
	elif 'x' in x:
		return np.nan
	elif len(x) == 1:
		if args.text_labels:
			return 'Neutral'
		else:
			return 0
	elif len(x) == 0:
		return np.nan
	else:
		if args.text_labels:
			return m[x[1]]
		else:
			return x[1]

def compute_stats(idf,tdf,pdf,fdf,bdf,
				  normalized_IPUS,normalized_turns,normalized_pauses,
				  normalized_ftos,normalized_backchannels):
	for data,name in [(fdf,'fto',),(tdf,'turn'), (idf,'IPU'),
			(pdf,'pause'),(bdf,'backhannel'),
			(normalized_ftos,'normalized-fto'),
			(normalized_turns,'normalized-turn'), 
			(normalized_IPUS,'normalized-IPU'),
			(normalized_pauses,'normalized-pause'),
			(normalized_backchannels,'normalized-backchannel')]:	
		try:
			data['stance_strength'] = data.next_stance.apply(get_strength)
			data['stance_polarity'] = data.next_stance.apply(get_polarity)
		except Exception:
			data['stance_strength'] = data.stance.apply(get_strength)
			data['stance_polarity'] = data.stance.apply(get_polarity)
		if not args.text_labels:
			stance_data = data.loc[~(data.stance_strength.isna()|(
					data.stance_polarity.isna())|data.stance_polarity.eq('x')
					)].reset_index(drop=True)
			stance_data.to_csv(os.path.join(args.outdir,f'{name}s-stance.csv'))

def plot_data(idf,tdf,pdf,fdf,bdf,normalized_IPUS,normalized_turns,
			  normalized_pauses,normalized_ftos,normalized_backchannels):
	sns.set_palette("colorblind")
	for frame in (idf,tdf,pdf,fdf,bdf,):#normalized_IPUS,normalized_turns,
		frame = frame.loc[abs(frame.duration-frame.duration.mean())<(
				args.n_sd*frame.duration.std())]
		try:
			frame['stance_strength'] = frame.next_stance.apply(get_strength)
			frame['stance_polarity'] = frame.next_stance.apply(get_polarity)

		except Exception:
			frame['stance_strength'] = frame.stance.apply(get_strength)
			frame['stance_polarity'] = frame.stance.apply(get_polarity)	
	
		frame = frame.loc[~(frame.stance_strength.isna()|(
				frame.stance_polarity.isna())|frame.stance_polarity.eq('x')
				)].reset_index(drop=True)

	data_tuples = [
			(fdf,'normalized-fto',[-5.5,5]),
			(idf,'normalized-IPU',[0,10]),
			(pdf,'normalized-pause',[-0.5,5]),
			(tdf,'normalized-turn',[0,15])]

	title_indices = ['FTO', 'IPU','Pause','Turn',]	
	# Loop through tuples and create violin plots
	lims = {}
	clips = {}
	titles = {}
	for i, (df, label,xlim) in enumerate(data_tuples):
		measure = label.split('-')[1]
		df['measure'] = measure
		lims[measure] = xlim
		titles[measure] = title_indices[i]

	df = pd.concat([a[0] for a in data_tuples])
	df = pd.melt(df, 
				 id_vars = ['measure','duration'],
				 value_vars = ['stance_strength','stance_polarity'],
				 value_name='stance-measure',var_name='stance-type')
	orders = ['None','Weak','Negative','Positive','Moderate','Neutral','Strong']
	g = sns.FacetGrid(df, row="stance-type", col='measure',
			hue="stance-measure", aspect=1.5, height=6, legend_out=True,
			palette='colorblind', sharey=False, sharex=False, hue_order=orders)
	g.map(sns.kdeplot, "duration", bw_adjust=.5, cut=0, fill=True,
			linewidth=0.25, alpha=0.25, legend=False)
	legends ={}
	for (row_val,col_val), ax in g.axes_dict.items():
		ax.set_xlim(lims[col_val])
		ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
		ax.tick_params(axis='both', which='both', labelsize=24, 
			labelfontfamily='Times New Roman')

		# Set the font and font size for axis labels
		if col_val == 'fto':
			ax.set_ylabel(' '.join(w[0].upper()+w[1:] for w in (
					row_val.split('_'))), fontsize=48, 
					fontdict={'family': 'Times New Roman'}, labelpad=24)
		else:
			ax.set_ylabel('')
		if row_val == 'stance_polarity':
			ax.set_xlabel('Duration (in s)', fontsize=36, 
					fontdict={'family': 'Times New Roman'},labelpad=24)
			ax.set_title('')
		else:
			ax.set_xlabel('')
			ax.set_title(titles[col_val],
					fontdict={'family':'Times New Roman', 'size': 72})
		if col_val =='turn':
			legends[row_val] = ax.legend(loc='center left',
					bbox_to_anchor=(1.1,0.5),
					prop={'family':'Times New Roman','size':48},
					ncol=1,columnspacing=0.5,
					handlelength=1,
					handletextpad=0.25,
					labelspacing=0.25,
					title_fontproperties={'family':'Times New Roman',
										  'size':48,},
					title=' '.join(w[0].upper()+w[1:] for w in (
						row_val.split('_')[1:])))
	plt.tight_layout()

	# Save the plot to SVG
	plt.savefig(os.path.join(args.outdir,args.plot_name))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--plot',action='store_true',
						help='whether to make plots')
	parser.add_argument('--plot_name',default='violinplots',
						help='filename for saving plot figures')
	parser.add_argument('--outdir','-o',
						help='dir to write files to')
	parser.add_argument('--overwrite',action='store_true',
						help='if the outdir exists, potentially overwrite files'
						)
	parser.add_argument('--load_stance',action='store_true',
						help='add stance annotations ')
	parser.add_argument('--flatten_stance',action='store_true',
						help='If flagged, define the stance of a turn as the '
							'highest strengh of all statnce annotations. If '
							'neutral and either positive or negative labels are'
							' present, prefer the non-neutral annotation.')
	parser.add_argument('--single_gender',action='store_true',
						help='if flagged, ignore interlocutor gender')
	parser.add_argument('--filter_bk',action='store_true')
	parser.add_argument('--sil_threshold',default=0.18,type=float) # 180ms
	parser.add_argument('--n_sd',default=2,type=int) # how many sd to normalize
	parser.add_argument('--debug',action='store_true',
						help='just try to load the textgrids')
	parser.add_argument('--text_labels',action='store_true',
						help='whether to make csv labels text')
	parser.add_argument('--single_channel_dir',
						help='path to single channel stance files',
						default='textgrids-fine/single-ch')
	parser.add_argument('--textgrid_dir',
						default='textgrids',
						help='path to textgrids with transcriptions '
								'(files titled Aligned*)')

	args = parser.parse_args()
	if (not os.path.exists(args.outdir)) or args.overwrite:
		if (not os.path.exists(args.outdir)):
			os.mkdir(args.outdir)
		(idf,tdf,pdf,fdf,bdf,normalized_IPUS,normalized_turns,normalized_pauses,
			normalized_ftos,normalized_backchannels) = load_data()
	else:
		tdf = pd.read_csv(os.path.join(args.outdir,'turns.csv'))
		idf = pd.read_csv(os.path.join(args.outdir,'IPUS.csv'))
		pdf = pd.read_csv(os.path.join(args.outdir,'pauses.csv'))
		fdf = pd.read_csv(os.path.join(args.outdir,'ftos.csv'))
		bdf = pd.read_csv(os.path.join(args.outdir,'backchannels.csv'))
		normalized_turns = pd.read_csv(
				os.path.join(args.outdir,'normalized_turns.csv'))
		normalized_IPUS = pd.read_csv(
				os.path.join(args.outdir,'normalized_IPUS.csv'))
		normalized_pauses = pd.read_csv(
				os.path.join(args.outdir,'normalized_pauses.csv'))
		normalized_ftos = pd.read_csv(
				os.path.join(args.outdir,'normalized_ftos.csv'))
		normalized_backchannels = pd.read_csv(
				os.path.join(args.outdir,'normalized_backchannels.csv'))

	compute_stats(idf,tdf,pdf,fdf,bdf,normalized_IPUS,
			normalized_turns,normalized_pauses,
			normalized_ftos,normalized_backchannels)
	if args.plot:
		plot_data(idf,tdf,pdf,fdf,bdf,normalized_IPUS,
			normalized_turns,normalized_pauses,
			normalized_ftos,normalized_backchannels)

