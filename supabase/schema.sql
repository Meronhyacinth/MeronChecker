-- MeronChecker database schema
-- Applied to Supabase project eojzxgypuivzazrlgciq.
-- Privacy: analysis text is never stored; only a SHA-256 fingerprint and non-sensitive metadata are retained.

create table public.analysis_metadata (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  model_version text not null,
  ai_likelihood numeric(5,2) check (ai_likelihood between 0 and 100),
  word_count integer not null check (word_count >= 0),
  input_sha256 text not null,
  retention_opt_in boolean not null default false
);

create index analysis_metadata_user_created_idx
  on public.analysis_metadata (user_id, created_at desc);

create table public.model_evaluations (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  model_version text not null,
  dataset_name text not null,
  accuracy numeric(6,5) not null check (accuracy between 0 and 1),
  precision_score numeric(6,5) not null check (precision_score between 0 and 1),
  recall_score numeric(6,5) not null check (recall_score between 0 and 1),
  f1_score numeric(6,5) not null check (f1_score between 0 and 1),
  notes text not null default ''
);

alter table public.analysis_metadata enable row level security;
alter table public.model_evaluations enable row level security;

grant select, insert on public.analysis_metadata to authenticated;
grant select on public.model_evaluations to anon, authenticated;

create policy "Users read their own analysis metadata"
on public.analysis_metadata for select to authenticated
using ((select auth.uid()) = user_id);

create policy "Users create their own analysis metadata"
on public.analysis_metadata for insert to authenticated
with check ((select auth.uid()) = user_id);

create policy "Public can read published model evaluations"
on public.model_evaluations for select to anon, authenticated
using (true);
