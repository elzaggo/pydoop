EXPORT_DIR = /tmp/pydoop_export
COPYRIGHT_OWNER = CRS4
NOTICE_TEMPLATE = notice_template.txt
COPYRIGHTER = copyrighter -n $(NOTICE_TEMPLATE) $(COPYRIGHT_OWNER)
# install copyrighter >=0.4.0 from ac-dc/tools/copyrighter

GENERATED_FILES = $(wildcard src/*_main.cpp) $(wildcard src/*.cc) DEFAULT_HADOOP_HOME pydoop/config.py
PY_V := $(shell python -c 'import sys; print "%d.%d" % sys.version_info[:2]')


.PHONY: all build build_py install install_py install_user install_user_py docs docs_py docs_put docs_view dist clean distclean uninstall_user logo favicon

all: build

build:
	python setup.py build

build_py:
	python setup.py build_py

install: build
	python setup.py install --skip-build

install_py: build_py
	python setup.py install --skip-build

install_user: build
	python setup.py install --skip-build --user

install_user_py: build_py
	python setup.py install --skip-build --user

logo: docs/_static/logo.png

favicon: docs/_static/favicon.ico

docs/_static/logo.png: logo/logo.svg
#	direct conversion to final size with inkscape does not look good
	inkscape -z -D -f $< -e logo/logo.png -w 800 # -b '#ffffff'
	convert -resize 200x logo/logo.png $@
	rm -f logo/logo.png

docs/_static/favicon.ico: logo/favicon.svg
	inkscape -z -D -f $< -e favicon-256.png -w 256 -h 256
	for i in 16 32 64 128; do \
	  convert favicon-256.png -resize $${i}x$${i} favicon-$${i}.png; \
	done
	convert favicon-16.png favicon-32.png favicon-64.png favicon-128.png $@
	rm -f favicon-*.png

docs: logo favicon
	make -C docs html

docs_py:
	make -C docs html

docs_put: docs
	rsync -avz --delete -e ssh docs/_build/html/ ${USER},pydoop@web.sourceforge.net:/home/project-web/pydoop/htdocs/docs/

docs_view: docs
	yelp docs/_build/html/index.html &

dist-svn: docs
	rm -rf $(EXPORT_DIR) && svn export . $(EXPORT_DIR)
	$(COPYRIGHTER) -r $(EXPORT_DIR)
	rm -rf $(EXPORT_DIR)/docs/*
	mv docs/_build/html $(EXPORT_DIR)/docs/
	cd $(EXPORT_DIR) && python setup.py sdist

dist: docs
	rm -rf $(EXPORT_DIR) && mkdir $(EXPORT_DIR) && cp -a * $(EXPORT_DIR)
	$(COPYRIGHTER) -r $(EXPORT_DIR)
	rm -rf $(EXPORT_DIR)/docs/*
	mv docs/_build/html $(EXPORT_DIR)/docs/
	(cd $(EXPORT_DIR) && python setup.py sdist) && mv -i $(EXPORT_DIR)/dist/pydoop-*.tar.gz . && rm -rf $(EXPORT_DIR)

clean:
	python setup.py clean
	rm -rf build
	rm -f $(GENERATED_FILES)
	rm -f docs/_static/logo.png docs/_static/favicon.ico
	make -C docs clean
	make -C examples/self_contained clean
	make -C examples/wordcount/c++ clean
	find . -regex '.*\(\.pyc\|\.pyo\|~\|\.so\|\.jar\|\.class\)' -exec rm -fv {} \;

distclean: clean
	rm -rf $(EXPORT_DIR)
	make -C examples/self_contained distclean

uninstall_user:
	rm -rf ~/.local/lib/python$(PY_V)/site-packages/pydoop*
